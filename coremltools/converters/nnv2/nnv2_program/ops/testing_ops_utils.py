import logging
import six
import numpy as np

import coremltools
import coremltools.converters.nnv2.converter as converter
from coremltools.converters.nnv2.nnv2_program.program import SsaProgram, SsaFunction
from coremltools.converters.nnv2.builtin_types.symbolic import is_symbolic
from coremltools.converters.nnv2._deps import HAS_TF

UNK_VARIADIC = '*s_unk'
UNK_SYM = 's_unk'

if HAS_TF:
    import tensorflow as tf

# backend --> target in convert function.
backend_to_convert_targets = {
        'nnv2': 'nnv2_proto',
        'nnv1': 'nnv1_proto',
        }

def _random_gen(shape, rand_min=0.0, rand_max=1.0, eps_from_int=0.0):
    """
        This helper function generates a random array of shape `shape`
        The range of generated numbers will be between (rand_min, rand_max].
        The value of generated numbers will be at least `eps_from_int` apart from integers.
    """

    elem = np.prod(shape)
    ret = []
    for _ in range(elem):
        while True:
            r = (rand_max - rand_min) * np.random.random() + rand_min
            if np.fabs(np.round(r)-r) > eps_from_int:
                ret.append(r)
                break
    ret = np.array(ret).reshape(shape)
    return ret


def ssa_fn(func):
    def wrapper(*args, **kwargs):
        prog = SsaProgram()
        with SsaFunction({}) as ssa_func:
            func(*args, **kwargs)
    return wrapper

def to_tuple(v):
    if not isinstance(v, (list, tuple)):
        return tuple([v])
    return tuple(v)

def is_close(expected, actual, atol=1e-04, rtol=1e-05):
    """
    expected, actual: np.array or python primitive (scalar)
    rtol: relative tolerance. See numpy.isclose.
    """

    close = np.isclose(expected, actual, atol=atol, rtol=rtol)
    if not np.all(close):
        diff = expected - actual
        num_not_close = np.sum(~close)
        msg = "Values differ by L1 norm: {}. Num entries not close: {}/{}"
        logging.error(msg.format(np.linalg.norm(diff, ord=1), num_not_close,
            expected.size))
        if num_not_close < 30:
            logging.error("Differing entries:")
            logging.error("Expected: {}".format(expected[~close]))
            logging.error("Actual: {}".format(actual[~close]))
            logging.error("Delta: {}".format(diff[~close]))
        return False
    return True

def compare_backend(proto, input_values, expected_outputs,
        use_cpu_only=False, atol=1e-04, rtol=1e-05):
    """
    Inputs:
        - proto: mlmodel proto.

        - input_values: str -> np.array. Keys must match those in
          input_placeholders.

        - expected_outputs: dict[str, np.array]. Required iff
          frontend_only == False

        - use_cpu_only: True/False.
    """
    mlmodel = coremltools.models.MLModel(proto, useCPUOnly=use_cpu_only)
    pred = mlmodel.predict(input_values, useCPUOnly=use_cpu_only)
    if not use_cpu_only:
        atol = min(atol * 100., 1e-1)
        rtol = min(rtol * 100., 1e-2)
    for o, expected in expected_outputs.items():
        assert is_close(expected, pred[o], atol, rtol), \
            'Output {} differs. useCPUOnly={}.\nInput={}, Expected={}, Output={}\n'.format(o, use_cpu_only, input_values, expected, pred[o])

def compare_shapes(proto, input_values, expected_outputs, use_cpu_only=False):
    """
    Inputs:
        - proto: mlmodel proto.

        - input_values: str -> np.array. Keys must match those in
          input_placeholders.

        - expected_outputs: dict[str, np.array].

        - use_cpu_only: True/False.
    """
    mlmodel = coremltools.models.MLModel(proto, useCPUOnly=use_cpu_only)
    pred = mlmodel.predict(input_values, useCPUOnly=use_cpu_only)
    for o, expected in expected_outputs.items():
        assert pred[o].shape == expected.shape


def get_core_ml_prediction(build, input_placeholders, input_values,
        use_cpu_only=False, backend='nnv1'):
    """
    Return predictions of the given model.
    """
    program = SsaProgram()
    with SsaFunction(input_placeholders) as ssa_func:
        output_vars = build(**ssa_func.inputs)
        if isinstance(output_vars, tuple):
            output_vars = list(output_vars)
        elif not isinstance(output_vars, list):
            output_vars = [output_vars]
        ssa_func.set_outputs(output_vars)
        program.add_function('main', ssa_func)

    convert_target = backend_to_convert_targets[backend]
    proto = converter.convert(program, convert_from='NitroSSA',
            convert_to=convert_target)
    model = coremltools.models.MLModel(proto, use_cpu_only)
    return model.predict(input_values, useCPUOnly=use_cpu_only)


def run_compare_builder(build, input_placeholders, input_values,
        expected_output_types=None, expected_outputs=None,
        use_cpu_only=False, frontend_only=False, backend='nnv1',
        atol=1e-04, rtol=1e-05):
    """
    Inputs:
        - build: python function taking input of Vars and returning Var or
          list[Var]. Each input argument in build must match a key in
          input_values / input_placeholders.

        - input_placeholders: str -> placeholder. It may not be an empty
                              dict as mlmodel doesn't support function with
                              no input.

        - expected_output_types: list[(shape, builtin_type)] or (shape,
          builtin_type).  None skips type inference validation.

        - expected_outputs: list[np.array] or np.array. Required iff
          frontend_only == False

        - frontend_only: True to test up to proto generation.
    """
    if not isinstance(expected_output_types, list):
        expected_output_types = [expected_output_types]

    if expected_outputs is not None and not isinstance(expected_outputs, list):
        expected_outputs = [expected_outputs]

    prog = SsaProgram()
    with SsaFunction(input_placeholders) as ssa_func:
        output_vars = build(**ssa_func.inputs)
        if isinstance(output_vars, tuple):
            output_vars = list(output_vars)
        elif not isinstance(output_vars, list):
            output_vars = [output_vars]
        ssa_func.set_outputs(output_vars)
        prog.add_function("main", ssa_func)

    # Validate type inference
    for out_var, s in zip(output_vars, expected_output_types):
        if out_var.dtype != s[-1]:
            raise ValueError('Output {} type: expect {}, got {}. Program:\n{}'.format(
                              out_var.name, s[-1], out_var.dtype, prog))
        if UNK_VARIADIC in s[:-1]:
            msg = 'Skip type checking for UNK_VARIADIC. Ouput_shape: {} vs expected shape: {}'
            logging.debug(msg.format(out_var.shape, s[:-1]))
            continue
        expected_shape = s[:-1]
        msg = 'Output {} shape: expect {}, got {}. Program:\n{}'.format(
              out_var.name, expected_shape, out_var.shape, prog)
        # No more variadic here.
        if len(out_var.shape) != len(expected_shape):   
            raise ValueError(msg)
        # replace UNK_SYM in out_var.shape.
        output_shape = [0 if es == UNK_SYM else os for os, es in zip(out_var.shape, expected_shape)]
        expected_shape = [0 if es == UNK_SYM else es for es in expected_shape]
        # convert float etc to int.
        output_shape = [i if is_symbolic(i) else int(i) for i in output_shape]
        expected_shape = [i if is_symbolic(i) else int(i) for i in expected_shape]
        if output_shape != expected_shape:
            raise ValueError(msg)

    convert_target = backend_to_convert_targets[backend]
    proto = converter.convert(prog,
                              convert_from="NitroSSA",
                              convert_to=convert_target)

    if frontend_only:
        return

    if expected_outputs:
        expected_outputs = {o.name: val for o, val in \
                zip(output_vars, expected_outputs)}

    compare_backend(proto=proto,
                    input_values=input_values,
                    expected_outputs=expected_outputs,
                    use_cpu_only=use_cpu_only,
                    atol=atol, rtol=rtol)

def tf_get_name(tfnode):
    if isinstance(tfnode, six.string_types):
        return tfnode.split(':')[0]
    return tfnode.name.split(':')[0]

def run_compare_tf(graph, placeholder_vals, output_nodes,
        use_cpu_only=False, frontend_only=False, backend='nnv1',
        atol=1e-04, rtol=1e-05, validate_shapes_only=False):
    """
    Inputs:
        - graph: tf.Graph

        - placeholder_vals: dict of tf.placeholder -> np.array/primitive.
          tf.placeholder must be part of `graph`.

        - output_nodes: tf.node or list[tf.node] representing outputs of
          `graph`.
    """
    if not isinstance(output_nodes, list):
        output_nodes = [output_nodes]

    # Convert TF graph.
    ph_list = [(k, v) for k, v in placeholder_vals.items()]
    input_values = {tf_get_name(k): v for k, v in placeholder_vals.items()}
    convert_target = backend_to_convert_targets[backend]
    proto = converter.convert(graph, convert_from="tensorflow",
            convert_to=convert_target,
            inputs=[tf_get_name(ph) for ph, _ in ph_list],
            outputs=[tf_get_name(node) for node in output_nodes])

    if frontend_only:
        return

    # Run TF
    output_vals_tf = tf.Session().run(output_nodes, feed_dict=placeholder_vals)
    expected_outputs = {tf_get_name(node): val for node, val in
            zip(output_nodes, output_vals_tf)}

    if validate_shapes_only:
        compare_shapes(proto, input_values, expected_outputs, use_cpu_only)
    else:
        compare_backend(proto, input_values, expected_outputs,
                        use_cpu_only, atol=atol, rtol=rtol)
