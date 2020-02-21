import numpy as np

from coremltools import models
from .nnv2_program.program import SsaProgram, SsaFunction
from coremltools.converters.nnv2 import converter
from .nnv2_program.ops.testing_ops_utils import backend_to_convert_targets


def build_main_program(inputs):
    def wrapper(main_block):
        program = SsaProgram()
        with SsaFunction(inputs) as func:
            func.set_outputs([main_block(*func.inputs.values())])
            program.add_function('main', func)
        return program

    return wrapper


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


def assert_model_is_valid(program, inputs, backend='nnv1',
        verbose=True):
    """
    Assert Core ML model is valid.

    Inputs:

    - input: str -> shape tuple. All program input names need to appear in str.
      shape tuple can only contain positive integers.
    """
    input_dict = dict()
    for name, shape in inputs.items():
        input_dict[name] = np.random.rand(*shape)
    convert_target = backend_to_convert_targets[backend]
    proto = converter.convert(program, convert_from='NitroSSA',
            convert_to=convert_target)
    model = models.MLModel(proto)
    assert model is not None
    if verbose:
        print(model)

    prediction = model.predict(input_dict)
    assert prediction is not None
    if verbose:
        print(prediction)
