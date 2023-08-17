#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause
import itertools

import copy
import os
from functools import partial
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pytest
from PIL import Image

import coremltools as ct
import coremltools.models.utils as coremltoolsutils
from coremltools._deps import _IS_MACOS
from coremltools.converters.mil.mil import Block, Function, Program
from coremltools.converters.mil.mil.passes.defs.preprocess import NameSanitizer as _NameSanitizer
from coremltools.converters.mil.mil.passes.defs.quantization import AbstractQuantizationPass
from coremltools.converters.mil.mil.passes.pass_registry import PASS_REGISTRY
from coremltools.proto import FeatureTypes_pb2 as ft

np.random.seed(10)

DTYPE_TO_FEATURE_TYPE_MAP: Dict[str, ft.ArrayFeatureType] = {
    "int32": ft.ArrayFeatureType.INT32,
    "fp32": ft.ArrayFeatureType.FLOAT32,
    "fp16": ft.ArrayFeatureType.FLOAT16,
}

# The minimum macOS version for an IOS target. For example, iOS16 target requires macOS13+.
IOS_TO_MINIMUM_MACOS_VERSION: Dict[ct.target, int] = {
    ct.target.iOS14: 11,
    ct.target.iOS15: 12,
    ct.target.iOS16: 13,
    ct.target.iOS17: 14,
}

hardcoded_einsum_equations: List[str] = [
    # hardcoded cases
    "abcd,adce->abce",
    "abc,cbd->abd",
    "bnqd,bnkd->bnqk",
    "abc,cd->abd",
    "abc,cde->abde",
    "btnh,bfnh->bnft",
    "bnft,btnh->bfnh",
    "abcd,cde->abe",
    "a b c d , a d c e -> a b c e",
]

einsum_equations: List[str] = hardcoded_einsum_equations + [
    # with-diagonal generic cases
    "jiii,ijjk->jk",
    "iji,ji->j",
    "jii,ijk->jk",
    "ijij,iij->ij",
    # no-diagonal generic cases
    "i,j->ij",  # outer product
    "a,a->a",  # batched outer product
    "ija,la->ijal",  # batched outer product
    "i,i->",  # inner product
    "ia,ia->a",  # batched inner product
    "ai,ia->a",  # batched inner product
    "abi,abi->ab",  # batched inner product
    "iab,iab->ab",  # batched inner product
    "abi,bai->ba",  # batched inner product
    "ij,j->i",  # matrix-vector multiplication
    "i,ij->j",  # vector-matrix multiplication
    "ai,ija->aj",  # batched vector-matrix multiplication
    "aibj,bi->jba",  # batched matrix-vector multiplication
    "ij,jk->ik",  # matrix multiplication
    "aij,ajk->iak",  # batched matrix multiplication
    "abij,abjk->abik",  # batched matrix multiplication
    "aijb,bajk->abik",  # batched matrix multiplication
    "ij,ij->",  # double-inner product
    "ij,ji->",  # double-inner product
    "aij,aij->a",  # batched double-inner product
    "ija,ija->a",  # batched double-inner product
    "ija,jia->a",  # batched double-inner product
    "aijb,ajbi->ab",  # batched double-inner product
    "aibj,cdij->cadb",  # batched double-inner product
    "ijk,lmj->iklm",  # 3rd-order tensor contraction
    "ijak,akl->aijl",  # batched 3rd-order tensor and matrix contraction
    # Generic with sum
    "ij,j->ij",
    "ij,kjl->j",
    "iijj,j->j",
]


def macos_compatible_with_deployment_target(minimum_deployment_target):
    if coremltoolsutils._is_macos():
        macos_major_version = coremltoolsutils._macos_version()[0]
        if macos_major_version < IOS_TO_MINIMUM_MACOS_VERSION[minimum_deployment_target]:
            return False
    return True

def _serialize_current_pytest(mlmodel):
    class_name = os.environ.get('PYTEST_CURRENT_TEST').split("::")[1].strip()
    test_name = "::".join(os.environ.get('PYTEST_CURRENT_TEST').split("::")[2:]).split("(call)")[0].strip()
    mlpackage_path = "/tmp/pytest_failures/{}/{}/model.mlpackage".format(class_name, test_name)
    Path(mlpackage_path).mkdir(parents=True, exist_ok=True)
    mlmodel.save(mlpackage_path)

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


def assert_model_is_valid(
    program,
    inputs,
    backend=("neuralnetwork", "fp32"),
    verbose=True,
    expected_output_shapes=None,
    minimum_deployment_target: ct.target = None,
):
    """
    Assert Core ML model is valid.

    Inputs:

    - input: str -> shape tuple. All program input names need to appear in str.
      shape tuple can only contain positive integers.
    """
    if minimum_deployment_target is not None:
        validate_minimum_deployment_target(minimum_deployment_target, backend)

    # Avoid circular import
    from coremltools.converters.mil.testing_reqs import ct

    input_dict = dict()
    for name, shape in inputs.items():
        input_dict[name] = np.random.rand(*shape)

    mlmodel = ct_convert(
        program,
        source="milinternal",
        convert_to=backend,
        compute_units=ct.ComputeUnit.CPU_ONLY,
        minimum_deployment_target=minimum_deployment_target,
    )
    assert mlmodel is not None

    if verbose:
        from coremltools.models.neural_network.printer import print_network_spec
        print_network_spec(mlmodel.get_spec(), style="coding")

    if _IS_MACOS and (not mlmodel.is_package or coremltoolsutils._macos_version() >= (12, 0)):
        prediction = mlmodel.predict(input_dict)
        assert prediction is not None
        if expected_output_shapes is not None:
            for out_name, out_shape in expected_output_shapes.items():
                assert out_name in prediction
                assert out_shape == prediction[out_name].shape, \
                        "{} != {}".format(out_shape, prediction[out_name].shape)

def assert_same_input_names(prog1, prog2, func_name="main"):
    # check the input keys
    prog1_input_keys = list(prog1[func_name].inputs.keys())
    prog2_input_keys = list(prog2[func_name].inputs.keys())
    assert prog1_input_keys == prog2_input_keys

    # check the input var name
    prog1_input_names = [x.name for x in list(prog1[func_name].inputs.values())]
    prog2_input_names = [x.name for x in list(prog2[func_name].inputs.values())]
    assert prog1_input_names == prog2_input_names


def assert_same_input_types(prog1, prog2, func_name="main"):
    prog1_input_types = [x.dtype for x in list(prog1[func_name].inputs.values())]
    prog2_input_types = [x.dtype for x in list(prog2[func_name].inputs.values())]
    assert prog1_input_types == prog2_input_types

def assert_same_output_names(prog1, prog2, func_name="main"):
    prog1_outputs = [o.name for o in prog1[func_name].outputs]
    prog2_outputs = [o.name for o in prog2[func_name].outputs]
    assert prog1_outputs == prog2_outputs

def assert_same_output_types(prog1: Program, prog2: Program, func_name: str = "main"):
    """
    Check ``prog1`` and ``prog2`` have the same output dtypes.
    """
    prog1_output_types = [o.dtype for o in prog1[func_name].outputs]
    prog2_output_types = [o.dtype for o in prog2[func_name].outputs]
    assert prog1_output_types == prog2_output_types

def assert_same_output_shapes(prog1, prog2, func_name="main"):
    prog1_output_shapes = [o.shape for o in prog1[func_name].outputs]
    prog2_output_shapes = [o.shape for o in prog2[func_name].outputs]
    assert prog1_output_shapes == prog2_output_shapes

def get_op_names_in_program(prog, func_name="main", skip_const_ops=True):
    """
    Return the operations names in prog[func_name],
    in the same order as they are stored (topological)
    """
    op_names_in_program = []
    for op in prog[func_name].operations:
        if skip_const_ops:
            if op.op_type == "const":
                continue
        op_names_in_program.append(op.name)
    return op_names_in_program


def get_op_types_in_block(block: Block, skip_const_ops: bool = True):
    """
    Return the operation types in block,
    in the same order as they are stored (topological)
    """
    op_types_in_block = []
    for op in block.operations:
        if skip_const_ops:
            if op.op_type == "const":
                continue
        op_types_in_block.append(op.op_type)
    return op_types_in_block


def get_op_types_in_program(prog: Program, func_name: str = "main", skip_const_ops: bool = True):
    """
    Return the operation types in prog[func_name],
    in the same order as they are stored (topological)
    If ``skip_const_ops = True``, const ops are not returned.
    """
    return get_op_types_in_block(prog[func_name], skip_const_ops)

def random_gen(
    shape,
    rand_min=0.0,
    rand_max=1.0,
    eps_from_int=0.0,
    allow_duplicate=True,
    dtype=np.float32,
):
    """
    This helper function generates a random array of shape `shape`
    The range of generated numbers will be between (rand_min, rand_max].
    The value of generated numbers will be at least `eps_from_int` apart from integers.
    If allow_duplicate is set to false, it is guaranteed that value generated are all different.
    Default data type is np.float32.
    """
    elem = np.prod(shape).astype(np.int32)

    # Since this function is extensively used as well for the fp16 precision models,
    # we make sure that the numerical value can be presented in fp16.
    gen_dtype = np.float16 if dtype == np.float32 else dtype
    ret = []
    for _ in range(elem):
        while True:
            r = gen_dtype((rand_max - rand_min) * np.random.random() + rand_min)
            if not allow_duplicate and r in ret:
                continue
            if np.issubdtype(gen_dtype, np.integer) or np.fabs(np.round(r) - r) > eps_from_int:
                ret.append(r)
                break
    ret = np.array(ret).reshape(shape)
    return ret.astype(dtype)


def ssa_fn(func):
    """
    Deprecated: use @mb.program()
    """

    def wrapper(*args, **kwargs):
        prog = Program()
        with Function({}) as ssa_func:
            func(*args, **kwargs)

    return wrapper


def to_tuple(v):
    if not isinstance(v, (list, tuple)):
        return tuple([v])
    return tuple(v)


def run_core_ml_predict(mlmodel, input_key_values):
    for k, v in input_key_values.items():
        if isinstance(v, Image.Image):
            continue
        elif not np.isscalar(v) and not v.shape == ():
            input_key_values[k] = v.astype(np.float32)
        else:
            input_key_values[k] = np.array([v], dtype=np.float32)
    return mlmodel.predict(input_key_values)

def _get_coreml_out_from_dict(out_dict, out_name):
    if out_name in out_dict:
        return out_dict[out_name]
    sanitized_out_name = _NameSanitizer._replace_invalid_char_with_underscore(out_name)
    if sanitized_out_name in out_dict:
        return out_dict[sanitized_out_name]
    else:
        raise KeyError(f"{out_name} output not found in Core ML outputs")

def _get_proto_output_shape(spec, out_name):
    sanitized_out_name = _NameSanitizer._replace_invalid_char_with_underscore(out_name)
    for coreml_o in spec.description.output:
        if coreml_o.name == sanitized_out_name:
            return coreml_o.type.multiArrayType.shape
    raise KeyError(f"{out_name} output not found in Core ML outputs")

def compare_backend(
    mlmodel,
    input_key_values,
    expected_outputs,
    dtype="fp32",
    atol=1e-04,
    rtol=1e-05,
    also_compare_shapes=True,
):
    """
    Inputs:
        - mlmodel: MLModel.

        - input_key_values: str -> np.array. Keys must match those in
          input_placeholders.

        - expected_outputs: dict[str, np.array]. Required iff
          frontend_only is False
    """
    if _IS_MACOS and (not mlmodel.is_package or coremltoolsutils._macos_version() >= (12, 0)):

        if dtype not in ["fp32", "fp16"]:
            raise ValueError("Unsupported dtype config")

        pred = run_core_ml_predict(mlmodel, input_key_values)
        if also_compare_shapes:
            compare_shapes(
                mlmodel,
                input_key_values,
                expected_outputs,
                pred=pred,
            )
        if mlmodel.compute_unit != ct.ComputeUnit.CPU_ONLY or (dtype == "fp16"):
            atol = max(atol * 100.0, 5e-1)
            rtol = max(rtol * 100.0, 5e-2)
        for o, expected in expected_outputs.items():
            coreml_out = _get_coreml_out_from_dict(pred, o)

            if isinstance(coreml_out, np.ndarray):
                np.testing.assert_allclose(coreml_out, expected, atol=atol, rtol=rtol)
            elif isinstance(coreml_out, dict):
                for k, v in coreml_out.items():
                    assert k in expected
                    assert expected[k] == v
            else:
                assert coreml_out == expected

        return pred
    return None


def compare_shapes(mlmodel, input_key_values, expected_outputs, pred=None):
    """
    Inputs:
        - mlmodel: MLModel.

        - input_key_values: str -> np.array or PIL.Image. Keys must match those in
          input_placeholders.

        - expected_outputs: dict[str, np.array].

        - pred: Prediction to use, if it has already been computed.
    """
    if _IS_MACOS:
        if not pred:
            pred = run_core_ml_predict(mlmodel, input_key_values)
        for o, expected in expected_outputs.items():
            coreml_out = _get_coreml_out_from_dict(pred, o)

            # output is dictionary (for classifier)
            if isinstance(coreml_out, dict) and isinstance(expected, dict):
                assert len(coreml_out) == len(expected)
                continue

            # output is numpy objects
            np_types = (np.generic, np.ndarray)
            if isinstance(coreml_out, np_types) and isinstance(expected, np_types):
                msg = "Output: {}. expected shape {} != actual shape {}".format(
                    o, expected.shape, coreml_out.shape
                )
                # Core ML does not support scalar as output
                # remove this special case when support is added
                if expected.shape == () and coreml_out.shape == (1,):
                    continue
                assert coreml_out.shape == expected.shape, msg

                # Validate the shape consistency across runtime returned values and
                # the output information in the mlprogram proto.
                spec = mlmodel.get_spec()
                if spec.WhichOneof("Type") == "mlProgram":
                    # The proto output and the runtime outputs are different for classifier
                    if spec.description.predictedFeatureName != "":
                        continue
                    proto_shape = _get_proto_output_shape(spec, o)
                    if proto_shape != []:
                        assert proto_shape == list(
                            coreml_out.shape
                        ), f"the output shape, for output named {o}, returned by the model is {coreml_out.shape} which does match with the shape present in the proto spec, which is {proto_shape}"
                continue

            # output is other types (for classifier)
            assert type(coreml_out) == type(expected)

def ct_convert(
    program,
    source="auto",
    inputs=None,
    outputs=None,
    classifier_config=None,
    minimum_deployment_target=None,
    convert_to=None,
    compute_precision=None,
    skip_model_load=False,
    converter=ct.convert,
    **kwargs,
):

    """
    Overloaded ct.convert function with the only difference being in the argument `convert_to`
    which in this overloaded call accepts a tuple of (target, dtype).
    Ex: ("neuralnetwork", "fp32"), ("mlprogram", "fp16")
    """

    if isinstance(converter, partial):
        raise ValueError("Partial function is not supported for function-parameter 'converter' since its keywords arguments could get overriden.")

    target, dtype = convert_to

    if dtype not in ["fp32", "fp16"]:
        raise ValueError("Unsupported dtype config")

    compute_precision = ct.precision.FLOAT16 if dtype == "fp16" else ct.precision.FLOAT32
    if target == "neuralnetwork":
        compute_precision = None

    mlmodel = converter(
                program,
                source=source,
                inputs=inputs,
                outputs=outputs,
                classifier_config=classifier_config,
                minimum_deployment_target=minimum_deployment_target,
                convert_to=target,
                compute_precision=compute_precision,
                skip_model_load=skip_model_load,
                **kwargs
    )

    if os.environ.get("DEBUG_SAVE_MLMODEL", "0") == "1":
        from coremltools.converters.mil.testing_utils import _serialize_current_pytest
        _serialize_current_pytest(mlmodel)

    return mlmodel

def get_core_ml_prediction(
    build, input_placeholders, input_values, backend, compute_unit=ct.ComputeUnit.CPU_ONLY
):
    """
    Return predictions of the given model.
    """
    minimum_deployment_target = backend.opset_version
    program = Program()
    with Function(input_placeholders, opset_version=minimum_deployment_target) as ssa_func:
        output_vars = build(**ssa_func.inputs)
        if isinstance(output_vars, tuple):
            output_vars = list(output_vars)
        elif not isinstance(output_vars, list):
            output_vars = [output_vars]
        ssa_func.set_outputs(output_vars)
        program.add_function("main", ssa_func)

    mlmodel = ct_convert(
        program,
        source="milinternal",
        convert_to=(backend.backend, backend.precision),
        compute_units=compute_unit,
        minimum_deployment_target=minimum_deployment_target,
    )
    return mlmodel.predict(input_values)


def apply_pass_and_basic_check(
    prog,
    pass_name,
    skip_output_name_check=False,
    skip_output_type_check=False,
    skip_input_name_check=False,
    skip_input_type_check=False,
):
    """
    Apply pass to the program
    """
    prev_prog = copy.deepcopy(prog)
    graph_pass = pass_name if isinstance(pass_name, AbstractQuantizationPass) else PASS_REGISTRY[pass_name]
    graph_pass(prog)
    block = prog.functions["main"]
    prev_block = prev_prog.functions["main"]
    if not skip_output_name_check:
        assert_same_output_names(prev_prog, prog)
    if not skip_output_type_check:
        assert_same_output_types(prev_prog, prog)
    assert_same_output_shapes(prev_prog, prog)

    if not skip_input_name_check:
        assert_same_input_names(prev_prog, prog)
    if not skip_input_type_check:
        assert_same_input_types(prev_prog, prog)
    return prev_prog, prev_block, block


def assert_prog_input_type(prog, expected_dtype_str, expected_name=None, index=0):
    block = prog.functions["main"]
    if expected_name is None:
        input_var = list(block.inputs.values())[index]
        assert input_var.is_tensor_or_scalar_of(dtype=expected_dtype_str)
    else:
        for input_var in block.inputs.values():
            if input_var.name == expected_name:
                assert input_var.is_tensor_or_scalar_of(dtype=expected_dtype_str)

def assert_spec_input_type(spec, expected_feature_type, expected_name=None, index=0):
    if expected_name is None:
        assert spec.description.input[index].type.multiArrayType.dataType == expected_feature_type
    else:
        for input in spec.description.input:
            if input.name == expected_name:
                assert input.type.multiArrayType.dataType == expected_feature_type

def assert_input_dtype(mlmodel, expected_type_str, expected_name=None, index=0):
    assert_prog_input_type(mlmodel._mil_program, expected_type_str,
                           expected_name=expected_name, index=index)
    assert_spec_input_type(mlmodel._spec, DTYPE_TO_FEATURE_TYPE_MAP[expected_type_str],
                           expected_name=expected_name, index=index)

def assert_spec_output_type(spec, expected_feature_type, expected_name=None, index=0):
    assert spec.description.output[index].type.multiArrayType.dataType == expected_feature_type
    if expected_name is not None:
        assert spec.description.output[index].name == expected_name

def assert_prog_output_type(prog, expected_dtype_str, expected_name=None, index=0):
    block = prog.functions["main"]
    output_var = block.outputs[index]
    assert output_var.is_tensor_or_scalar_of(dtype=expected_dtype_str)
    if expected_name is not None:
        assert output_var.name == expected_name

def assert_output_dtype(mlmodel, expected_type_str, expected_name=None, index=0):
    assert_prog_output_type(mlmodel._mil_program, expected_type_str,
                            expected_name=expected_name, index=index)
    assert_spec_output_type(mlmodel._spec, DTYPE_TO_FEATURE_TYPE_MAP[expected_type_str],
                            expected_name=expected_name, index=index)

def random_gen_input_feature_type(input_desc):
    if input_desc.type.WhichOneof("Type") == "multiArrayType":
        shape = [s for s in input_desc.type.multiArrayType.shape]
        if input_desc.type.multiArrayType.dataType == ft.ArrayFeatureType.FLOAT32:
            dtype = np.float32
        elif input_desc.type.multiArrayType.dataType == ft.ArrayFeatureType.INT32:
            dtype = np.int32
        elif input_desc.type.multiArrayType.dataType == ft.ArrayFeatureType.FLOAT16:
            dtype = np.float16
        elif input_desc.type.multiArrayType.dataType == ft.ArrayFeatureType.FLOAT64:
            dtype = np.float64
        else:
            raise ValueError("unsupported type")
        return np.random.rand(*shape).astype(dtype)
    elif input_desc.type.WhichOneof("Type") == "imageType":
        if input_desc.type.imageType.colorSpace in (ft.ImageFeatureType.BGR, ft.ImageFeatureType.RGB):
            shape = [3, input_desc.type.imageType.height, input_desc.type.imageType.width]
            x = np.random.randint(low=0, high=256, size=shape)
            return Image.fromarray(np.transpose(x, [1, 2, 0]).astype(np.uint8))
        elif input_desc.type.imageType.colorSpace == ft.ImageFeatureType.GRAYSCALE:
            shape = [input_desc.type.imageType.height, input_desc.type.imageType.width]
            x = np.random.randint(low=0, high=256, size=shape)
            return Image.fromarray(x.astype(np.uint8), 'L')
        elif input_desc.type.imageType.colorSpace == ft.ImageFeatureType.GRAYSCALE_FLOAT16:
            shape = (input_desc.type.imageType.height, input_desc.type.imageType.width)
            x = np.random.rand(*shape)
            return Image.fromarray(x.astype(np.float32), 'F')
        else:
            raise ValueError("unrecognized image type")
    else:
        raise ValueError('unsupported type')


def gen_input_shapes_einsum(equation: str, dynamic: bool, backend: Tuple[str, str]):
    equation = equation.replace(" ", "")
    left = equation.split("->")[0]
    var_descs = left.split(",")
    converter_shapes = {}
    shapes = {}
    cur_default_shape = 2
    for symbol in itertools.chain.from_iterable(var_descs):
        if symbol not in shapes:
            shapes[symbol] = cur_default_shape
            if dynamic:
                converter_shapes[symbol] = ct.RangeDim(
                    default=cur_default_shape,
                    upper_bound=cur_default_shape if backend[0] == "mlprogram" else -1,
                )
            else:
                converter_shapes[symbol] = cur_default_shape
            cur_default_shape += 1
    var_shapes = [[shapes[symbol] for symbol in var_desc] for var_desc in var_descs]
    converted_shapes = [ct.TensorType(shape=[converter_shapes[symbol] for symbol in var_desc], dtype=np.float32)
                        for var_desc in var_descs]
    return var_shapes, converted_shapes


def verify_prediction(mlmodel, multiarray_type=None):
    spec = mlmodel._spec
    input_dict = {}
    for input_desc in spec.description.input:
        input_dict[input_desc.name] = random_gen_input_feature_type(input_desc)
        if multiarray_type is not None:
            input_dict[input_desc.name] = input_dict[input].astype(multiarray_type)
    mlmodel.predict(input_dict)

def assert_spec_input_image_type(spec, expected_feature_type):
    assert spec.description.input[0].type.imageType.colorSpace == expected_feature_type

def assert_spec_output_image_type(spec, expected_feature_type):
    assert spec.description.output[0].type.imageType.colorSpace == expected_feature_type

def assert_cast_ops_count(mlmodel, expected_count):
    block = mlmodel._mil_program.functions["main"]
    assert len(block.find_ops(op_type="cast")) == expected_count

def assert_ops_in_mil_program(mlmodel, expected_op_list):
    assert expected_op_list == get_op_types_in_program(mlmodel._mil_program)


def validate_minimum_deployment_target(
    minimum_deployment_target: ct.target, backend: Tuple[str, str]
):
    """
    Validates the minimum deployment target based on backend and macOS version. Only used in tests.
    """
    if minimum_deployment_target >= ct.target.iOS15 and backend[0] != "mlprogram":
        pytest.skip("IOS15+ target only compatible with mlprogram.")
    if not macos_compatible_with_deployment_target(minimum_deployment_target):
        pytest.skip(
            f"IOS{minimum_deployment_target} target is not runnable on this macOS {coremltoolsutils._macos_version()}"
        )
