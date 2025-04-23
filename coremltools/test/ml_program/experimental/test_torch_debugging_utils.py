#  Copyright (c) 2024, Apple Inc. All rights reserved.

from typing import Callable, Tuple

import numpy as np
import pytest
import torch

import coremltools as ct
from coremltools import proto
from coremltools._deps import _HAS_TORCH_EXPORT_API
from coremltools.models.ml_program.experimental.torch.debugging_utils import (
    TorchExportMLModelComparator,
    TorchScriptMLModelComparator,
    _convert_and_retrieve_jit_module_mapping,
    convert_and_retrieve_op_mapping,
    get_stack_frame_infos,
    inline_and_annotate_module,
)


class TestTorchMapping:
    @staticmethod
    def _get_simple_model() -> torch.nn.Module:
        class Sub(torch.nn.Module):
            def forward(self, x, y):
                return x - y

        class Add(torch.nn.Module):
            def forward(self, x, y):
                return x + y

        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.add = Add()
                self.sub = Sub()

            def forward(self, x, y):
                a = self.add(x, y)
                b = self.sub(x, y)
                return (a, b)

        model = Model()
        model.eval()
        return model

    def test_convert_and_retrieve_jit_module_mapping(self):
        model = TestTorchMapping._get_simple_model()
        input1 = torch.full((1, 10), 1, dtype=torch.float)
        input2 = torch.full((1, 10), 2, dtype=torch.float)
        example_inputs = (
            input1,
            input2,
        )
        traced_model = torch.jit.trace(model, example_inputs=example_inputs)

        _, module_mapping = _convert_and_retrieve_jit_module_mapping(
            model=traced_model,
            inputs=[
                ct.TensorType(name="x", shape=example_inputs[0].shape, dtype=np.float16),
                ct.TensorType(name="y", shape=example_inputs[0].shape, dtype=np.float16),
            ],
            minimum_deployment_target=ct.target.iOS16,
        )

        # The root module is identified by an empty string ('') for the module name
        # and 0 for the call sequence, represented as a tuple key ('', 0).
        root_module = module_mapping.get(("", 0), None)
        assert root_module is not None, "Expected to find 'root' module in module_mapping"

        add_module = module_mapping.get(("add", 0), None)
        assert add_module is not None, "Expected to find 'add' module in module_mapping"

        sub_module = module_mapping.get(("sub", 0), None)
        assert sub_module is not None, "Expected to find 'sub' module in module_mapping"

    def test_module_inlining_and_naming(self):
        model = TestTorchMapping._get_simple_model()
        input1 = torch.full((1, 10), 1, dtype=torch.float)
        input2 = torch.full((1, 10), 2, dtype=torch.float)
        example_inputs = (
            input1,
            input2,
        )
        traced_model = torch.jit.trace(model, example_inputs=example_inputs)
        name_prefix = "test"
        inline_and_annotate_module(
            model=traced_model,
            name_prefix=name_prefix,
        )

        graph = traced_model.graph
        for node in graph.nodes():
            kind = node.kind()
            assert kind != "prim::CallMethod", "Expected all prim::CallMethod nodes to be inlined"
            for output in node.outputs():
                assert output.debugName().startswith(
                    name_prefix
                ), f"Expected debug name to start with {name_prefix}"

    def test_module_call_stack(self):
        model = TestTorchMapping._get_simple_model()
        input1 = torch.full((1, 10), 1, dtype=torch.float)
        input2 = torch.full((1, 10), 2, dtype=torch.float)
        example_inputs = (
            input1,
            input2,
        )
        traced_model = torch.jit.trace(model, example_inputs=example_inputs)
        name_prefix = "test"
        annotator = inline_and_annotate_module(
            model=traced_model,
            name_prefix=name_prefix,
        )

        expected_call_sequence = [("add", 0), ("sub", 0), ("", 0)]
        call_sequence = []
        for module_key in annotator.module_call_stack:
            call_sequence.append(module_key)

        assert (
            expected_call_sequence == call_sequence
        ), f"Expected module call sequence ({expected_call_sequence}) but got ({call_sequence})"

    def test_source_to_target_op_mapping(self):
        model = TestTorchMapping._get_simple_model()
        input1 = torch.full((1, 10), 1, dtype=torch.float)
        input2 = torch.full((1, 10), 2, dtype=torch.float)
        example_inputs = (
            input1,
            input2,
        )
        traced_model = torch.jit.trace(model, example_inputs=example_inputs)

        model, mapping = convert_and_retrieve_op_mapping(
            model=traced_model,
            inputs=[
                ct.TensorType(name="x", shape=example_inputs[0].shape, dtype=np.float16),
                ct.TensorType(name="y", shape=example_inputs[0].shape, dtype=np.float16),
            ],
            minimum_deployment_target=ct.target.iOS16,
        )

        spec = model.get_spec()
        function = spec.mlProgram.functions["main"]
        for operation in function.block_specializations[function.opset].operations:
            source_nodes = mapping.get_source_nodes_for_operation(operation=operation)
            assert (
                len(source_nodes) == 1
            ), f"Expected to find one source node for operation {operation} but found {source_nodes}."

        sub_ops = []
        add_ops = []
        for source_op, target_ops in mapping.node_to_operations_map.items():
            if source_op.kind == "aten::sub":
                sub_ops.extend(target_ops)

            if source_op.kind == "aten::add":
                add_ops.extend(target_ops)

        assert (
            len(sub_ops) == 1 and sub_ops[0].type == "sub"
        ), f"Expected to find one 'sub' op but found ({sub_ops})"
        assert (
            len(add_ops) == 1 and add_ops[0].type == "add"
        ), f"Expected to find one 'add' op but found ({add_ops})"

    @pytest.mark.parametrize("export_method", ["jit", "export"])
    def test_stack_frame_infos(self, export_method: str):
        model = TestTorchMapping._get_simple_model()
        input1 = torch.full((1, 10), 1, dtype=torch.float)
        input2 = torch.full((1, 10), 2, dtype=torch.float)
        example_inputs = (
            input1,
            input2,
        )

        traced_model = None
        if export_method == "jit":
            traced_model = torch.jit.trace(model, example_inputs)
        elif export_method == "export":
            traced_model = torch.export.export(model, example_inputs)

        assert traced_model is not None

        model, mapping = convert_and_retrieve_op_mapping(
            model=traced_model,
            inputs=[
                ct.TensorType(name="x", shape=example_inputs[0].shape, dtype=np.float16),
                ct.TensorType(name="y", shape=example_inputs[0].shape, dtype=np.float16),
            ],
            minimum_deployment_target=ct.target.iOS16,
        )

        for node, ops in mapping.node_to_operations_map.items():
            if len(ops) > 0:
                frames = get_stack_frame_infos(node=node)
                assert (
                    frames is not None and len(frames) > 0
                ), f"Expected non-empty stack frame information for {node}. "


class TestTorchModelComparator:
    @staticmethod
    def transform_model_spec(
        model: ct.models.MLModel,
        update_fn: Callable[[proto.MIL_pb2.Operation], None],
    ) -> ct.models.MLModel:
        def clone_spec(
            spec: proto.Model_pb2.Model,
        ) -> proto.Model_pb2.Model:
            spec_class = spec.__class__
            new_spec = spec_class()
            new_spec.CopyFrom(spec)
            return new_spec

        def transform_spec(
            model: proto.Model_pb2.Model,
        ):
            program_spec = model.mlProgram
            for _, function_spec in program_spec.functions.items():
                block_spec = function_spec.block_specializations.get(function_spec.opset, None)
                if block_spec is None:
                    continue

                for op in block_spec.operations:
                    update_fn(op)

        spec = model.get_spec()
        if spec.WhichOneof("Type") != "mlProgram":
            raise ValueError("transform_model_spec only supports ML Program.")

        cloned_spec = clone_spec(spec)
        transform_spec(cloned_spec)

        return ct.models.MLModel(
            model=cloned_spec,
            weights_dir=model.weights_dir,
        )

    @staticmethod
    def _get_simple_model() -> torch.nn.Module:
        class Mul(torch.nn.Module):
            def forward(self, x, y):
                return x * y

        class Sub(torch.nn.Module):
            def forward(self, x, y):
                return x - y

        class Add(torch.nn.Module):
            def forward(self, x, y):
                return x + y

        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.add = Add()
                self.sub = Sub()
                self.mul = Mul()

            def forward(self, x, y):
                a = self.add(x, y)
                b = self.sub(x, y)
                c = self.mul(x, y)
                return (a, b, c)

        model = Model()
        model.eval()
        return model

    @staticmethod
    def create_torch_script_model_comparator(
        inputs: Tuple[torch.Tensor],
    ) -> TorchScriptMLModelComparator:
        model = TestTorchModelComparator._get_simple_model()
        comparator = TorchScriptMLModelComparator(
            model=model,
            example_inputs=inputs,
            inputs=[
                ct.TensorType(name="x", shape=inputs[0].shape, dtype=np.float16),
                ct.TensorType(name="y", shape=inputs[1].shape, dtype=np.float16),
            ],
            minimum_deployment_target=ct.target.iOS16,
            compute_units=ct.ComputeUnit.ALL,
        )

        return comparator

    @staticmethod
    def create_torch_export_model_comparator(
        inputs: Tuple[torch.Tensor],
    ) -> TorchExportMLModelComparator:
        model = TestTorchModelComparator._get_simple_model()
        exported_model = torch.export.export(model, inputs)
        comparator = TorchExportMLModelComparator(
            model=exported_model,
            inputs=[
                ct.TensorType(name="x", shape=inputs[0].shape, dtype=np.float16),
                ct.TensorType(name="y", shape=inputs[1].shape, dtype=np.float16),
            ],
            minimum_deployment_target=ct.target.iOS16,
            compute_units=ct.ComputeUnit.ALL,
        )

        return comparator

    @staticmethod
    def compare_outputs(
        op: proto.MIL_pb2.Operation,
        reference_output: np.array,
        target_output: np.array,
    ) -> bool:

        return np.allclose(reference_output, target_output, atol=0.01)

    @staticmethod
    def update_op_type(
        from_type: str,
        to_type: str,
    ) -> None:
        def _update_op_type(op: proto.MIL_pb2.Operation):
            if op.type == from_type:
                op.type = to_type

        return _update_op_type

    @pytest.mark.asyncio
    async def test_torch_script_numeric_issues(self):
        input1 = torch.full((1, 10), 1, dtype=torch.float)
        input2 = torch.full((1, 10), 2, dtype=torch.float)
        inputs = (
            input1,
            input2,
        )

        comparator = TestTorchModelComparator.create_torch_script_model_comparator(inputs=inputs)
        target_model = comparator.target_model
        # Transform the target model by replacing mul operations with add operations.
        # This transformation intentionally introduces numerical discrepancies in the model's behavior.
        transformed_model = TestTorchModelComparator.transform_model_spec(
            target_model,
            update_fn=TestTorchModelComparator.update_op_type("mul", "add"),
        )

        comparator._set_target_model(target_model=transformed_model)
        modules = await comparator.find_failing_modules(
            inputs=inputs,
            compare_outputs=TestTorchModelComparator.compare_outputs,
        )

        assert len(modules) == 1 and modules[0].key == (
            "mul",
            0,
        ), f"Expected to find 'mul' module but found ({modules})"

        # Transform the target model by replacing sub operations with add operations.
        # This transformation intentionally introduces numerical discrepancies in the model's behavior.
        transformed_model = TestTorchModelComparator.transform_model_spec(
            target_model,
            update_fn=TestTorchModelComparator.update_op_type("sub", "add"),
        )

        comparator._set_target_model(target_model=transformed_model)
        modules = await comparator.find_failing_modules(
            inputs=inputs,
            compare_outputs=TestTorchModelComparator.compare_outputs,
        )

        assert len(modules) == 1 and modules[0].key == (
            "sub",
            0,
        ), f"Expected to find 'sub' module but found ({modules})"

        # Transform the target model by replacing add operations with sub operations.
        # This transformation intentionally introduces numerical discrepancies in the model's behavior.
        transformed_model = TestTorchModelComparator.transform_model_spec(
            target_model,
            update_fn=TestTorchModelComparator.update_op_type("add", "sub"),
        )

        comparator._set_target_model(target_model=transformed_model)
        modules = await comparator.find_failing_modules(
            inputs=inputs,
            compare_outputs=TestTorchModelComparator.compare_outputs,
        )

        assert len(modules) == 1 and modules[0].key == (
            "add",
            0,
        ), f"Expected to find 'add' module but found ({modules})"

    @pytest.mark.asyncio
    @pytest.mark.skipif(not _HAS_TORCH_EXPORT_API, reason="This test requires PyTorch Export APIs.")
    async def test_torch_export_numeric_issues(self):
        input1 = torch.full((1, 10), 1, dtype=torch.float)
        input2 = torch.full((1, 10), 2, dtype=torch.float)
        inputs = (
            input1,
            input2,
        )

        comparator = TestTorchModelComparator.create_torch_export_model_comparator(inputs=inputs)
        target_model = comparator.target_model
        # Transform the target model by replacing mul operations with add operations.
        # This transformation intentionally introduces numerical discrepancies in the model's behavior.
        transformed_model = TestTorchModelComparator.transform_model_spec(
            target_model,
            update_fn=TestTorchModelComparator.update_op_type("mul", "add"),
        )

        comparator._set_target_model(target_model=transformed_model)
        ops = await comparator.find_failing_ops(
            inputs=inputs,
            compare_outputs=TestTorchModelComparator.compare_outputs,
        )

        assert (
            len(ops) == 1 and str(ops[0].target) == "aten.mul.Tensor"
        ), f"Expected to find a 'mul' op but found ({ops})"

        # Transform the target model by replacing sub operations with add operations.
        # This transformation intentionally introduces numerical discrepancies in the model's behavior.
        transformed_model = TestTorchModelComparator.transform_model_spec(
            target_model,
            update_fn=TestTorchModelComparator.update_op_type("sub", "add"),
        )

        comparator._set_target_model(target_model=transformed_model)
        ops = await comparator.find_failing_ops(
            inputs=inputs,
            compare_outputs=TestTorchModelComparator.compare_outputs,
        )

        assert (
            len(ops) == 1 and str(ops[0].target) == "aten.sub.Tensor"
        ), f"Expected to find a 'sub' op but found ({ops})"

        # Transform the target model by replacing add operations with sub operations.
        # This transformation intentionally introduces numerical discrepancies in the model's behavior.
        transformed_model = TestTorchModelComparator.transform_model_spec(
            target_model,
            update_fn=TestTorchModelComparator.update_op_type("add", "sub"),
        )

        comparator._set_target_model(target_model=transformed_model)
        ops = await comparator.find_failing_ops(
            inputs=inputs,
            compare_outputs=TestTorchModelComparator.compare_outputs,
        )

        assert (
            len(ops) == 1 and str(ops[0].target) == "aten.add.Tensor"
        ), f"Expected to find a 'add' op but found ({ops})"
