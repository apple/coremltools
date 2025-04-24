#  Copyright (c) 2023, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import re
from typing import Callable

import numpy as np
import pytest

import coremltools as ct
from coremltools import proto
from coremltools.converters.mil import Builder as mb
from coremltools.models.ml_program.experimental.debugging_utils import (
    MLModelComparator,
    MLModelInspector,
    MLModelValidator,
    compute_snr_and_psnr,
    skip_op_by_type,
)


def get_simple_program():
    @mb.program(
        input_specs=[
            mb.TensorSpec(shape=(1, 2, 3, 4)),
        ]
    )
    def prog(x):
        y = mb.const(val=1.2, name="y")
        x = mb.add(x=x, y=y, name="add")
        perm = mb.const(val=[0, 2, 3, 1], name="perm")
        x = mb.transpose(x=x, perm=perm, name="transpose_1")
        x = mb.square(x=x, name="output_0")
        x = mb.tanh(x=x, name="output_1")
        x = mb.transpose(x=x, perm=perm, name="transpose_2")
        return x

    return prog


def compute_ground_truth_answer(input):
    x = input + 1.2
    x = np.transpose(x, axes=[0, 2, 3, 1])
    square = x * x
    tanh = np.tanh(square)
    return {"output_0": square, "output_1": tanh}


class TestMLModelInspector:
    def test_error_handling(self):
        prog = get_simple_program()
        mlmodel = ct.convert(prog, convert_to="neuralnetwork")

        with pytest.raises(ValueError, match="MLModelInspector only supports ML program"):
            MLModelInspector(model=mlmodel)

        mlmodel = ct.convert(prog, convert_to="mlprogram")
        with pytest.raises(
            TypeError,
            match='The "compute_units" parameter must be of type "ComputeUnit"',
        ):
            MLModelInspector(model=mlmodel, compute_units="xyz")

        with pytest.raises(
            TypeError,
            match='The "function_name" parameter must be of type "str"',
        ):
            MLModelInspector(model=mlmodel, function_name=1)

        function_name = "xyz"
        with pytest.raises(
            ValueError,
            match=re.escape("Missing function for name : xyz"),
        ):
            MLModelInspector(model=mlmodel, function_name=function_name)

    def test_output_names(self):
        prog = get_simple_program()
        mlmodel = ct.convert(prog, convert_to="mlprogram", compute_precision=ct.precision.FLOAT32)
        inspector = MLModelInspector(model=mlmodel)
        expected_output_names = [
            "transpose_2",
            "output_1",
            "output_0",
            "transpose_1",
            "perm",
            "add",
            "y",
        ]

        assert (
            inspector.output_names == expected_output_names
        ), f"Expected output names do not match. Expected: {expected_output_names}, but got: {inspector.output_names}"

        assert list(inspector.output_name_to_op_map.keys()) == list(
            reversed(expected_output_names)
        ), f"Expected output names do not match. Expected: {reversed(expected_output_names)}, but got: {inspector.output_names}"

    @pytest.mark.asyncio
    async def test_output_value_is_not_none(self):
        prog = get_simple_program()
        mlmodel = ct.convert(prog, convert_to="mlprogram", compute_precision=ct.precision.FLOAT32)
        inspector = MLModelInspector(model=mlmodel, compute_units=ct.ComputeUnit.CPU_ONLY)
        inputs = {"x": np.random.rand(1, 2, 3, 4)}
        output_names = []
        async for name, value in inspector.inspect(inputs=inputs, ignore_const_ops=False):
            assert value is not None
            output_names.append(name)

        assert len(output_names) == len(inspector.output_names)
        for name1, name2 in zip(output_names, inspector.output_names):
            assert name1 == name2

    @pytest.mark.asyncio
    @pytest.mark.parametrize("num_predict_intermediate_outputs", [1, 2])
    async def test_output_value_is_valid(self, num_predict_intermediate_outputs: int):
        prog = get_simple_program()
        mlmodel = ct.convert(prog, convert_to="mlprogram", compute_precision=ct.precision.FLOAT32)
        inspector = MLModelInspector(model=mlmodel, compute_units=ct.ComputeUnit.CPU_ONLY)
        input = np.random.rand(1, 2, 3, 4)
        expected_outputs = compute_ground_truth_answer(input=input)
        async for name, value in inspector.inspect(
            inputs={"x": input},
            output_names=["output_0", "output_1"],
            num_predict_intermediate_outputs=num_predict_intermediate_outputs,
        ):
            np.testing.assert_allclose(value, expected_outputs[name], atol=0.2)

    @pytest.mark.asyncio
    async def test_invalid_output_name(self):
        prog = get_simple_program()
        mlmodel = ct.convert(prog, convert_to="mlprogram", compute_precision=ct.precision.FLOAT32)
        inspector = MLModelInspector(model=mlmodel, compute_units=ct.ComputeUnit.CPU_ONLY)
        input = np.random.rand(1, 2, 3, 4)
        expected_output_names = [
            "transpose_2",
            "output_1",
            "output_0",
            "transpose_1",
            "perm",
            "add",
            "y",
        ]

        message = f"Invalid output names (['output_3']). Available output names are: {expected_output_names}"
        with pytest.raises(
            ValueError,
            match=re.escape(message),
        ):
            async for name, value in inspector.inspect(
                inputs={"x": input}, output_names=["output_0", "output_1", "output_3"]
            ):
                pass


class TestMLModelValidator:
    @staticmethod
    def _get_test_program_with_div():
        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 2)), mb.TensorSpec(shape=(1, 2))])
        def prog(x, y):
            x1 = mb.add(x=x, y=y, name="add1")
            x2 = mb.real_div(x=x1, y=y, name="div1")
            x3 = mb.real_div(x=x1, y=y, name="div2")
            x4 = mb.add(x=x2, y=x3, name="add2")
            x5 = mb.add(x=x3, y=x4, name="add3")
            x6 = mb.mul(x=x4, y=x5, name="mul")
            return x6

        return prog

    @staticmethod
    def _get_test_program_with_softmax():
        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 2)), mb.TensorSpec(shape=(1, 2))])
        def prog(x, y):
            x1 = mb.add(x=x, y=y, name="add")
            x2 = mb.softmax(x=x1, axis=0, name="softmax")
            x3 = mb.sub(x=x2, y=y, name="sub")
            return x3

        return prog

    @pytest.mark.asyncio
    @pytest.mark.parametrize("num_predict_intermediate_outputs", [1, 2])
    async def test_inf_outputs(self, num_predict_intermediate_outputs: int):
        prog = TestMLModelValidator._get_test_program_with_div()
        mlmodel = ct.convert(prog, convert_to="mlprogram", compute_precision=ct.precision.FLOAT16)
        validator = MLModelValidator(
            model=mlmodel,
            compute_units=ct.ComputeUnit.CPU_ONLY,
            num_predict_intermediate_outputs=num_predict_intermediate_outputs,
        )
        x = np.ones((1, 2), dtype=np.float32)
        y = np.ones((1, 2), dtype=np.float32)
        ops = await validator.find_failing_ops_with_infinite_output(inputs={"x": x, "y": y})
        assert (
            len(ops) == 0
        ), f"Expected to find no ops with infinite values, but found {len(ops)} op(s) with inf value(s)"
        y = np.zeros((1, 2), dtype=np.float32)
        ops = await validator.find_failing_ops_with_infinite_output(inputs={"x": x, "y": y})
        assert (
            len(ops) == 2
        ), f"Expected to find 2 ops with infinite values, but found {len(ops)} op(s) with inf value(s)"
        for op in ops:
            assert op.type == "real_div"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("num_predict_intermediate_outputs", [1, 2])
    async def test_nan_outputs(self, num_predict_intermediate_outputs: int):
        prog = TestMLModelValidator._get_test_program_with_softmax()
        mlmodel = ct.convert(prog, convert_to="mlprogram", compute_precision=ct.precision.FLOAT16)
        validator = MLModelValidator(
            model=mlmodel,
            compute_units=ct.ComputeUnit.CPU_ONLY,
            num_predict_intermediate_outputs=num_predict_intermediate_outputs,
        )
        x = np.full((1, 2), np.finfo(np.float16).max / 20.0, dtype=np.float32)
        y = np.full((1, 2), np.finfo(np.float16).max / 20.0, dtype=np.float32)
        ops = await validator.find_failing_ops_with_nan_output(inputs={"x": x, "y": y})
        assert (
            len(ops) == 0
        ), f"Expected to find no ops with nan values, but found {len(ops)} op(s) with nan value(s)"
        neg_inf_float16 = np.float16(-np.inf)
        x = np.full((1, 2), neg_inf_float16, dtype=np.float32)
        y = np.full((1, 2), neg_inf_float16, dtype=np.float32)
        # softmax of -inf will result in a nan
        ops = await validator.find_failing_ops_with_nan_output(inputs={"x": x, "y": y})
        assert (
            len(ops) == 1
        ), f"Expected to find 1 op with nan values, but found {len(ops)} op(s) with nan value(s)"
        assert ops[0].type == "softmax"


class TestMLModelComparator:
    @staticmethod
    def _get_test_program():
        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 2)), mb.TensorSpec(shape=(1, 2))])
        def prog(x, y):
            x1 = mb.add(x=x, y=y, name="add")
            x2 = mb.sub(x=x, y=y, name="sub")
            x3 = mb.mul(x=x1, y=x2, name="mul1")
            x4 = mb.mul(x=x1, y=x2, name="mul2")
            return (x3, x4)

        return prog

    async def verify_failing_ops(
        self,
        reference_precision: ct.precision,
        reference_compute_units: ct.ComputeUnit,
        target_precision: ct.precision,
        target_compute_units: ct.ComputeUnit,
        skip_op: Callable[[proto.MIL_pb2.Operation], bool],
        num_predict_intermediate_outputs: int,
    ):
        prog = TestMLModelComparator._get_test_program()
        reference_model = ct.convert(
            prog,
            convert_to="mlprogram",
            compute_precision=reference_precision,
            compute_units=reference_compute_units,
        )
        target_model = ct.convert(
            prog,
            convert_to="mlprogram",
            compute_precision=target_precision,
            compute_units=target_compute_units,
        )

        comparator = MLModelComparator(
            reference_model=reference_model,
            target_model=target_model,
            num_predict_intermediate_outputs=num_predict_intermediate_outputs,
        )
        x = np.ones((1, 2), dtype=np.float32)
        y = np.zeros((1, 2), dtype=np.float32)

        ops = await comparator.find_failing_ops(
            inputs={"x": x, "y": y},
            compare_outputs=lambda op, x, y: compute_snr_and_psnr(x, y)[1] > 60.0,
        )
        assert (
            len(ops) == 0
        ), f"Expected to find no ops that failed comparison, but found {len(ops)} op(s)"

        spec = target_model.get_spec()
        function = spec.mlProgram.functions["main"]
        block = function.block_specializations[function.opset]
        # Change ``sub`` ops to ``add`` ops
        for op in block.operations:
            if op.type == "sub":
                op.type = "add"

        target_model = ct.models.MLModel(
            model=spec,
            weights_dir=reference_model.weights_dir,
            compute_units=target_compute_units,
        )

        comparator = MLModelComparator(
            reference_model=reference_model,
            target_model=target_model,
        )
        x = np.ones((1, 2), dtype=np.float32)
        y = np.ones((1, 2), dtype=np.float32)

        ops = await comparator.find_failing_ops(
            inputs={"x": x, "y": y},
            compare_outputs=lambda op, x, y: compute_snr_and_psnr(x, y)[1] > 60.0,
            skip_op=skip_op,
        )
        assert (
            len(ops) == 1
        ), f"Expected to find a single op that failed comparison, but found {len(ops)} op(s)"
        assert ops[0].type == "sub", f"Expected to find the modified sub op, but found {ops[0]}"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("target_precision", [ct.precision.FLOAT32, ct.precision.FLOAT16])
    @pytest.mark.parametrize(
        "target_compute_units", [compute_unit for compute_unit in ct.ComputeUnit]
    )
    @pytest.mark.parametrize("num_predict_intermediate_outputs", [1, 2])
    async def test_failing_ops(
        self,
        target_precision: ct.precision,
        target_compute_units: ct.ComputeUnit,
        num_predict_intermediate_outputs: int,
    ):
        # - Reference model: Set to use CPU_ONLY compute units and FLOAT32 precision.
        # - Target model: Uses the provided target_precision and target_compute_units
        await self.verify_failing_ops(
            reference_precision=ct.precision.FLOAT32,
            reference_compute_units=ct.ComputeUnit.CPU_ONLY,
            target_precision=target_precision,
            target_compute_units=target_compute_units,
            skip_op=skip_op_by_type,
            num_predict_intermediate_outputs=num_predict_intermediate_outputs,
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("target_precision", [ct.precision.FLOAT32, ct.precision.FLOAT16])
    @pytest.mark.parametrize(
        "target_compute_units", [compute_unit for compute_unit in ct.ComputeUnit]
    )
    @pytest.mark.parametrize("num_predict_intermediate_outputs", [1, 2])
    async def test_skipping_ops(
        self,
        target_precision: ct.precision,
        target_compute_units: ct.ComputeUnit,
        num_predict_intermediate_outputs: int,
    ):
        # Skip mul ops, the failing op search should still work even if some ops are skipped
        # or missing from the target model.
        def skip_mul_op(op: proto.MIL_pb2.Operation) -> bool:
            return op.type == "mul"

        # - Reference model: Set to use CPU_ONLY compute units and FLOAT32 precision.
        # - Target model: Uses the provided target_precision and target_compute_units
        await self.verify_failing_ops(
            reference_precision=ct.precision.FLOAT32,
            reference_compute_units=ct.ComputeUnit.CPU_ONLY,
            target_precision=target_precision,
            target_compute_units=target_compute_units,
            skip_op=skip_mul_op,
            num_predict_intermediate_outputs=num_predict_intermediate_outputs,
        )
