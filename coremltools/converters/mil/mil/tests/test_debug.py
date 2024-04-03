#  Copyright (c) 2023, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import itertools
import os
import tempfile

import pytest
import numpy as np

import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.debugging_utils import extract_submodel
from coremltools.converters.mil.mil import get_new_symbol
from coremltools.converters.mil.mil.types.symbolic import is_symbolic
from coremltools.converters.mil.testing_utils import get_op_types_in_program

def get_simple_program():
    @mb.program(input_specs=[mb.TensorSpec(shape=(1, 2, 3, 4)),])
    def prog(x):
        x = mb.add(x=x, y=1.2, name="add")
        x = mb.transpose(x=x, perm=[0, 2, 3, 1])
        x = mb.square(x=x, name="output_0")
        x = mb.tanh(x=x, name="output_1")
        x = mb.transpose(x=x, perm=[0, 2, 3, 1])
        return x

    return prog

def compute_ground_truth_answer(input):
    x = input + 1.2
    x = np.transpose(x, axes=[0, 2, 3, 1])
    square = x * x
    tanh = np.tanh(square)
    return {"output_0": square, "output_1":tanh}

class TestExtractSubModel:

    def test_extract_submodel_error_handling(self):
        prog = get_simple_program()
        mlmodel = ct.convert(prog, convert_to="neuralnetwork")

        invalid_outputs = set()
        with pytest.raises(ValueError, match="outputs must be of type list/tuple. Got <class 'set'>"):
            extract_submodel(mlmodel, outputs=invalid_outputs)

        invalid_outputs = ["output_1", 1]
        with pytest.raises(ValueError, match="outputs must be a list of str. Got element 1 with type <class 'int'>."):
            extract_submodel(mlmodel, outputs=invalid_outputs)

        invalid_outputs = ["output_1", "output_1"]
        with pytest.raises(ValueError, match="outputs must be a list of unique elements. 'output_1' occurs 2 times"):
            extract_submodel(mlmodel, outputs=invalid_outputs)

        invalid_outputs = ["error"]
        with pytest.raises(ValueError, match="outputs \['error'\] not found in the function."):
            extract_submodel(mlmodel, outputs=invalid_outputs)

        model_dir = tempfile.TemporaryDirectory()
        mlmodel_path = os.path.join(model_dir.name, "model.mlmodel")
        mlmodel.save(mlmodel_path)
        mlmodel = ct.models.MLModel(mlmodel_path)
        with pytest.raises(ValueError, match="NeuralNetwork model loaded from the disk is not supported by the extract_submodel util"):
            extract_submodel(mlmodel, outputs=["output_0", "output_1"])

    def test_extract_submodel_symbolic_input(self):
        """
        Input graph:
        x -> sin ---> sub -> output_1
                  |
                  v
                 mul -> tan -> output_2

        If x has symbolic shape, then the subgraph mil -> tan should also have symbolic shape
        """
        @mb.program(input_specs=[mb.TensorSpec(shape=(1, get_new_symbol()))])
        def prog(x):
            sin = mb.sin(x=x, name="sin")
            sub = mb.sub(x=sin, y=1.5, name="sub")
            mul = mb.mul(x=sin, y=1.2, name="mul")
            tan = mb.tan(x=mul, name="tan")
            return sub, tan
        model = ct.convert(prog, convert_to="neuralnetwork")
        submodel = extract_submodel(model, outputs=["tan"], inputs=["mul"])
        func = submodel._mil_program.functions["main"]

        input = list(func.inputs.values())[0]
        assert input.shape[0] == 1
        assert is_symbolic(input.shape[1])

        output = func.outputs[0]
        assert output.shape[0] == 1
        assert is_symbolic(output.shape[1])

    def test_extract_submodel_complex(self):
        """
        Input graph:
        x -> sin ------> sub -> output_1
              |      |
              v      v
        y -> add -> mul -> tan -> realdiv -> output_2
        """
        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 2)), mb.TensorSpec(shape=(1, 2))])
        def prog(x, y):
            sin = mb.sin(x=x, name="sin")
            add = mb.add(x=sin, y=y, name="add")
            sub = mb.sub(x=sin, y=1.5, name="sub")
            mul = mb.mul(x=sin, y=add, name="mul")
            tan = mb.tan(x=mul, name="tan")
            realdiv = mb.real_div(x=tan, y=4.7, name="realdiv")
            return sub, realdiv
        model = ct.convert(prog, convert_to="neuralnetwork")

        """
        Case 1:
        inputs = None
        outputs = [sin, mul]

        Output graph:
        x -> sin ------> output_1
              |      |
              v      v
        y -> add -> mul -> output_2
        """
        submodel = extract_submodel(model, outputs=["sin", "mul"])
        assert get_op_types_in_program(submodel._mil_program) == ["sin", "add", "mul"]

        """
        Case 2:
        inputs = None
        outputs = [sin, add]

        Output graph:
        x -> sin -> output_1
              |
              v
        y -> add -> output_2
        """
        submodel = extract_submodel(model, outputs=["sin", "add"])
        assert get_op_types_in_program(submodel._mil_program) == ["sin", "add"]

        """
        Case 3:
        inputs = None
        outputs = [mul]

        Output graph:
        x -> sin -----
              |      |
              v      v
        y -> add -> mul -> output_1
        """
        submodel = extract_submodel(model, outputs=["mul"])
        assert get_op_types_in_program(submodel._mil_program) == ["sin", "add", "mul"]

        """
        Case 4:
        inputs = None
        outputs = [sin, sub]

        Output graph:
        x -> sin -> sub -> output_2
              |
              V
           output_1
        y
        """
        submodel = extract_submodel(model, outputs=["sin", "sub"])
        assert get_op_types_in_program(submodel._mil_program) == ["sin", "sub"]

        """
        Case 5:
        inputs = [x, y]
        outputs = [mul]

        Output graph:
        x -> sin -----
              |      |
              v      v
        y -> add -> mul -> output_1
        """
        submodel = extract_submodel(model, outputs=["mul"], inputs=["x", "y"])
        assert get_op_types_in_program(submodel._mil_program) == ["sin", "add", "mul"]

        """
        Case 6:
        inputs = [mul]
        outputs = [tan]

        mul -> tan -> output_1
        """
        submodel = extract_submodel(model, outputs=["tan"], inputs=["mul"])
        assert get_op_types_in_program(submodel._mil_program) == ["tan"]

        """
        Case 7:
        inputs = [sin, add]
        outputs = [sub, mul]

        sin ------> sub -> output_1
                |
                v
        add -> mul -> output_2
        """
        submodel = extract_submodel(model, outputs=["sub", "mul"], inputs=["sin", "add"])
        assert get_op_types_in_program(submodel._mil_program) == ["sub", "mul"]

        """
        Case 8 (Negative):
        inputs = [sin]
        outputs = [mul]

        mul not reachable merely through sin
        """
        with pytest.raises(ValueError, match="output mul not reachable from inputs"):
            submodel = extract_submodel(model, outputs=["mul"], inputs=["sin"])

        """
        Case 9 (Negative):
        inputs = [mul]
        outputs = [sin]

        sin not reachable merely through sin
        """
        with pytest.raises(ValueError, match="output sin not reachable from inputs"):
            submodel = extract_submodel(model, outputs=["sin"], inputs=["mul"])

    @pytest.mark.parametrize(
        "compute_unit",
        [
            ct.ComputeUnit.ALL,
            ct.ComputeUnit.CPU_ONLY,
        ],
    )
    def test_extract_submodel_neuralnetwork(self, compute_unit):
        prog = get_simple_program()
        model = ct.convert(prog, convert_to="neuralnetwork", compute_units=compute_unit)
        submodel = extract_submodel(model, outputs=["output_0", "output_1"])

        # check that the submodel retains the same backend
        assert submodel.get_spec().WhichOneof("Type") == "neuralNetwork"

        # check that the submodel retains the same compute unit
        assert submodel.compute_unit == compute_unit

        # check the subgraph
        assert get_op_types_in_program(submodel._mil_program) == ["add", "transpose", "square", "tanh"]

        # check the numerical outputs
        coreml_in = np.random.rand(1, 2, 3, 4)
        coreml_out = submodel.predict({"x": coreml_in})
        gt = compute_ground_truth_answer(coreml_in)
        assert len(coreml_out) == len(gt)
        for k, v in gt.items():
            np.testing.assert_allclose(v, coreml_out[k], atol=0.2)

    @pytest.mark.parametrize(
        "compute_unit, store_to_disk",
        itertools.product(
            [
                ct.ComputeUnit.ALL,
                ct.ComputeUnit.CPU_ONLY,
            ],
            [True, False],
        )
    )
    def test_extract_submodel_mlprogram(self, compute_unit, store_to_disk):
        prog = get_simple_program()
        model = ct.convert(
                    prog,
                    convert_to="mlprogram",
                    compute_units=compute_unit,
                    compute_precision=ct.precision.FLOAT32
                )

        if store_to_disk:
            model_dir = tempfile.TemporaryDirectory()
            mlmodel_path = os.path.join(model_dir.name, "model.mlpackage")
            model.save(mlmodel_path)
            model = ct.models.MLModel(mlmodel_path, compute_units=compute_unit)

        submodel = extract_submodel(model, outputs=["output_0", "output_1"])

        # check that the submodel retains the same backend
        assert submodel.get_spec().WhichOneof("Type") == "mlProgram"

        # check that the submodel retains the same compute unit
        assert submodel.compute_unit == compute_unit

        # check the subgraph
        assert get_op_types_in_program(submodel._mil_program) == ["add", "transpose", "square", "tanh"]

        # check the numerical outputs
        coreml_in = np.random.rand(1, 2, 3, 4)
        coreml_out = submodel.predict({"x": coreml_in})
        gt = compute_ground_truth_answer(coreml_in)
        assert len(coreml_out) == len(gt)
        for k, v in gt.items():
            np.testing.assert_allclose(v, coreml_out[k], atol=0.2)
