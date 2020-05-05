import logging
import numpy as np

import coremltools
from coremltools import converters as converter
from coremltools.converters.nnv2 import converter as _converter

from coremltools.converters.nnv2.nnv2_program.program import SsaProgram, SsaFunction

converter = converter
_converter = _converter


def assert_op_count_match(program, expect, op=None, verbose=False):
    """
    Assert number of ops match expected number. If op is not specified,
    Count total number of ops and match with expect.
    """
    if verbose:
        print(program)

    count = 0
    for _, func in program.functions.items():
        for o in func.operations:
            if not op:
                count += 1
            elif o.op_type.lower() == op.lower():
                count += 1
        np.testing.assert_equal(count, expect)


def assert_model_is_valid(program, inputs, backend='nnv1_proto',
        verbose=True, expected_output_shapes=None):
    """
    Assert Core ML model is valid.

    Inputs:

    - input: str -> shape tuple. All program input names need to appear in str.
      shape tuple can only contain positive integers.
    """
    input_dict = dict()
    for name, shape in inputs.items():
        input_dict[name] = np.random.rand(*shape)
    proto = _converter._convert(program,
                                convert_from='NitroSSA',
                                convert_to=backend)
    model = coremltools.models.MLModel(proto)
    assert model is not None
    if verbose:
        from coremltools.models.neural_network.printer import print_network_spec
        print_network_spec(proto, style='coding')

    prediction = model.predict(input_dict, useCPUOnly=True)
    assert prediction is not None
    if expected_output_shapes is not None:
        for out_name, out_shape in expected_output_shapes.items():
            assert out_name in prediction
            assert out_shape == prediction[out_name].shape


def assert_same_output_names(prog1, prog2, func_name='main'):
    prog1_outputs = [o.name for o in prog1[func_name].outputs]
    prog2_outputs = [o.name for o in prog2[func_name].outputs]
    assert prog1_outputs == prog2_outputs


def assert_same_output_shapes(prog1, prog2, func_name='main'):
    prog1_output_shapes = [o.shape for o in prog1[func_name].outputs]
    prog2_output_shapes = [o.shape for o in prog2[func_name].outputs]
    assert prog1_output_shapes == prog2_output_shapes


def get_op_types_in_program(prog, func_name='main', skip_const_ops=True):
    """
    Return the operation types in prog[func_name],
    in the same order as they are stored (topological)
    """
    op_types_in_program = []
    for op in prog[func_name].operations:
        if skip_const_ops:
            if op.op_type == 'const':
                continue
        op_types_in_program.append(op.op_type)
    return op_types_in_program


def random_gen(shape, rand_min=0.0, rand_max=1.0, eps_from_int=0.0, allow_duplicate=True, dtype=np.float32):
    """
    This helper function generates a random array of shape `shape`
    The range of generated numbers will be between (rand_min, rand_max].
    The value of generated numbers will be at least `eps_from_int` apart from integers.
    If allow_duplicate is set to false, it is guaranteed that value generated are all different.
    Default data type is np.float32.
    """

    elem = np.prod(shape)
    ret = []
    for _ in range(elem):
        while True:
            r = dtype((rand_max - rand_min) * np.random.random() + rand_min)
            if not allow_duplicate and r in ret:
                continue
            if np.fabs(np.round(r) - r) > eps_from_int:
                ret.append(r)
                break
    ret = np.array(ret).reshape(shape)
    return ret.astype(dtype)


def ssa_fn(func):
    """
    Deprecated: use @cb.program()
    """
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
        logging.error(msg.format(np.sum(np.abs(diff)), num_not_close,
                                 expected.size))
        if num_not_close < 30:
            logging.error("Differing entries:")
            logging.error("Expected: {}".format(expected[~close]))
            logging.error("Actual: {}".format(actual[~close]))
            logging.error("Delta: {}".format(diff[~close]))
        return False
    return True


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
        assert is_close(expected, pred[o], atol, rtol), msg.format(
            o, use_cpu_only, input_key_values, expected, pred[o])


def compare_shapes(
        proto, input_key_values, expected_outputs, use_cpu_only=False):
    """
    Inputs:
        - proto: MLModel proto.

        - input_key_values: str -> np.array. Keys must match those in
          input_placeholders.

        - expected_outputs: dict[str, np.array].

        - use_cpu_only: True/False.
    """
    model = coremltools.models.MLModel(proto, useCPUOnly=use_cpu_only)
    pred = model.predict(input_key_values, useCPUOnly=use_cpu_only)
    for o, expected in expected_outputs.items():
        msg = 'Output: {}. expected shape {} != actual shape {}'.format(o,
                expected.shape, pred[o].shape)
        assert pred[o].shape == expected.shape, msg


def get_core_ml_prediction(
        build, input_placeholders, input_values,
        use_cpu_only=False, backend='nnv1_proto'):
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

    proto = _converter._convert(program,
                                convert_from='NitroSSA',
                                convert_to=backend)
    model = coremltools.models.MLModel(proto, use_cpu_only)
    return model.predict(input_values, useCPUOnly=use_cpu_only)
