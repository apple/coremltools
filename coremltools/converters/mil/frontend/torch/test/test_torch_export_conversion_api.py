#  Copyright (c) 2023, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import itertools

import pytest

from coremltools._deps import _HAS_EXECUTORCH, _HAS_TORCH_EXPORT_API

if not _HAS_TORCH_EXPORT_API:
    pytest.skip(allow_module_level=True, reason="torch.export is required")

USE_EDGE_DIALECT = [False]
if _HAS_EXECUTORCH:
    USE_EDGE_DIALECT.append(True)

import torch

from coremltools.converters.mil import testing_reqs
from coremltools.converters.mil.mil.scope import ScopeSource

from .testing_utils import TorchBaseTest, TorchFrontend

backends = testing_reqs.backends
compute_units = testing_reqs.compute_units


class TestTorchExportConversionAPI(TorchBaseTest):
    @pytest.mark.parametrize(
        "compute_unit, backend, use_edge_dialect, dynamic",
        itertools.product(compute_units, backends, USE_EDGE_DIALECT, (True, False)),
    )
    def test_mul(self, compute_unit, backend, use_edge_dialect, dynamic):
        class MulModule(torch.nn.Module):
            def forward(self, input, other):
                return input * other

        dynamic_shapes = None
        if dynamic:
            dim0 = torch.export.Dim("dim0")
            dim1 = torch.export.Dim("dim1", min=1, max=3)
            dynamic_shapes = {
                "input": {0: dim0, 1: dim1},
                "other": {0: dim0, 1: dim1},
            }

        _, coreml_model, _, _, _, _ = self.run_compare_torch(
            [(3, 2), (3, 2)],
            MulModule(),
            compute_unit=compute_unit,
            backend=backend,
            frontend=TorchFrontend.EXIR,
            use_edge_dialect=use_edge_dialect,
            torch_export_dynamic_shapes=dynamic_shapes,
        )

        mil_program = coreml_model._mil_program
        mul = mil_program.functions["main"].find_ops(op_type="mul")[0]

        stack_trace = mul.scopes[ScopeSource.EXIR_STACK_TRACE][0]
        assert stack_trace.split("\n")[-2].strip() == "return input * other"

        if use_edge_dialect:
            debug_handle = mul.scopes[ScopeSource.EXIR_DEBUG_HANDLE][0]
            assert isinstance(debug_handle, int)

            debug_handle_to_ops_mapping = mil_program.construct_debug_handle_to_ops_mapping()
            assert debug_handle_to_ops_mapping.keys() == {debug_handle}

            ops = debug_handle_to_ops_mapping[debug_handle]
            index_mul = 0
            indices_const = ()
            indices_cast = ()
            if backend[1] == "fp32":
                assert len(ops) == 1
                index_mul = 0
            else:
                # fp16 introduces additional io casts
                # each cast introduces 1 const to store destination dtype
                assert len(ops) == 7
                index_mul = 4
                indices_const = (0, 1, 5)
                indices_cast = (2, 3, 6)
            assert ops[index_mul] == [
                {"Type": "Program"},
                {"Type": "Function", "Name": "main"},
                {"Type": "Block"},
                {"Type": "Operation", "Operator": "mul", "Output": mul.outputs[0].name},
            ]
            for index_const_cast in indices_const + indices_cast:
                assert ops[index_const_cast][:-1] == [
                    {"Type": "Program"},
                    {"Type": "Function", "Name": "main"},
                    {"Type": "Block"},
                ]
            for index_const in indices_const:
                assert ops[index_const][-1]["Operator"] == "const"
            for index_cast in indices_cast:
                assert ops[index_cast][-1]["Operator"] == "cast"

    @pytest.mark.parametrize(
        "compute_unit, backend, use_edge_dialect, dynamic",
        itertools.product(compute_units, backends, USE_EDGE_DIALECT, (True, False)),
    )
    def test_linear(self, compute_unit, backend, use_edge_dialect, dynamic):
        class LinearModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, 3)

            def forward(self, arg):
                return self.linear(arg)

        dynamic_shapes = None
        if dynamic:
            batch_dim = torch.export.Dim("batch_dim")
            dynamic_shapes = {"arg": {0: batch_dim}}

        _, coreml_model, _, _, _, _ = self.run_compare_torch(
            [(3, 3)],
            LinearModule(),
            compute_unit=compute_unit,
            backend=backend,
            frontend=TorchFrontend.EXIR,
            use_edge_dialect=use_edge_dialect,
            torch_export_dynamic_shapes=dynamic_shapes,
        )

        mil_program = coreml_model._mil_program
        linear = mil_program.functions["main"].find_ops(op_type="linear")[0]

        stack_trace = linear.scopes[ScopeSource.EXIR_STACK_TRACE][0]
        assert stack_trace.split("\n")[-2].strip() == "return self.linear(arg)"

        if use_edge_dialect:
            debug_handle = linear.scopes[ScopeSource.EXIR_DEBUG_HANDLE][0]
            assert isinstance(debug_handle, int)

            debug_handle_to_ops_mapping = mil_program.construct_debug_handle_to_ops_mapping()
            assert debug_handle_to_ops_mapping.keys() == {debug_handle}

            ops = debug_handle_to_ops_mapping[debug_handle]
            index_linear = 0
            indices_const = ()
            indices_cast = ()
            if backend[1] == "fp32":
                assert len(ops) == 3
                index_linear = 2
                indices_const = (0, 1)
            else:
                # fp16 introduces additional io casts
                # each cast introduces 1 const to store destination dtype
                assert len(ops) == 7
                index_linear = 4
                indices_const = (0, 1, 2, 5)
                indices_cast = (3, 6)
            assert ops[index_linear] == [
                {"Type": "Program"},
                {"Type": "Function", "Name": "main"},
                {"Type": "Block"},
                {"Type": "Operation", "Operator": "linear", "Output": linear.outputs[0].name},
            ]
            for index_const_cast in indices_const + indices_cast:
                assert ops[index_const_cast][:-1] == [
                    {"Type": "Program"},
                    {"Type": "Function", "Name": "main"},
                    {"Type": "Block"},
                ]
            for index_const in indices_const:
                assert ops[index_const][-1]["Operator"] == "const"
            for index_cast in indices_cast:
                assert ops[index_cast][-1]["Operator"] == "cast"

    @pytest.mark.parametrize(
        "compute_unit, backend, use_edge_dialect, dynamic",
        itertools.product(compute_units, backends, USE_EDGE_DIALECT, (True, False)),
    )
    def test_add(self, compute_unit, backend, use_edge_dialect, dynamic):
        if dynamic:
            pytest.skip(
                "https://github.com/apple/coremltools/issues/2307 "
                "torch.export has not settled the dynamism from 0/1 static shape yet"
            )

        class AddModule(torch.nn.Module):
            def forward(self, x, y):
                z = x + y
                z = z + x
                z = z + x
                z = z + z
                return z

        dynamic_shapes = None
        if dynamic:
            dim0 = torch.export.Dim("dim0", min=1)
            dynamic_shapes = {"x": {0: dim0}, "y": {0: dim0}}

        _, coreml_model, _, _, _, _ = self.run_compare_torch(
            [(1,), (1,)],
            AddModule(),
            compute_unit=compute_unit,
            backend=backend,
            frontend=TorchFrontend.EXIR,
            use_edge_dialect=use_edge_dialect,
            torch_export_dynamic_shapes=dynamic_shapes,
        )

        mil_program = coreml_model._mil_program
        adds = mil_program.functions["main"].find_ops(op_type="add")

        stack_traces = [add.scopes[ScopeSource.EXIR_STACK_TRACE][0] for add in adds]
        source_codes = [
            "z = x + y",
            "z = z + x",
            "z = z + x",
            "z = z + z",
        ]
        for i, stack_trace in enumerate(stack_traces):
            assert stack_trace.split("\n")[-2].strip() == source_codes[i]

        if use_edge_dialect:
            debug_handles = [add.scopes[ScopeSource.EXIR_DEBUG_HANDLE][0] for add in adds]
            for debug_handle in debug_handles:
                assert isinstance(debug_handle, int)

            debug_handle_to_ops_mapping = mil_program.construct_debug_handle_to_ops_mapping()
            assert debug_handle_to_ops_mapping.keys() == set(debug_handles)

            for add_index, debug_handle in enumerate(debug_handles):
                add = adds[add_index]
                ops = debug_handle_to_ops_mapping[debug_handle]
                index_add = 0
                indices_const = ()
                indices_cast = ()
                if backend[1] == "fp32":
                    assert len(ops) == 1
                    index_add = 0
                else:
                    # fp16 introduces additional io casts
                    # each cast introduces 1 const to store destination dtype
                    ADD_INDEX_TO_NUM_OPS = {0: 5, 1: 1, 2: 1, 3: 3}
                    ADD_INDEX_TO_OP_INDEX = {0: -1, 1: 0, 2: 0, 3: 0}
                    assert len(ops) == ADD_INDEX_TO_NUM_OPS[add_index]
                    index_add = ADD_INDEX_TO_OP_INDEX[add_index]
                    if add_index == 0:
                        indices_const = (0, 1)
                        indices_cast = (2, 3)
                    elif add_index == 3:
                        indices_const = (1,)
                        indices_cast = (2,)
                assert ops[index_add] == [
                    {"Type": "Program"},
                    {"Type": "Function", "Name": "main"},
                    {"Type": "Block"},
                    {"Type": "Operation", "Operator": "add", "Output": add.outputs[0].name},
                ]
                for index_const_cast in indices_const + indices_cast:
                    assert ops[index_const_cast][:-1] == [
                        {"Type": "Program"},
                        {"Type": "Function", "Name": "main"},
                        {"Type": "Block"},
                    ]
                for index_const in indices_const:
                    assert ops[index_const][-1]["Operator"] == "const"
                for index_cast in indices_cast:
                    assert ops[index_cast][-1]["Operator"] == "cast"

    @pytest.mark.parametrize(
        "compute_unit, backend, use_edge_dialect, dynamic",
        itertools.product(compute_units, backends, USE_EDGE_DIALECT, (True, False)),
    )
    def test_add_mul(self, compute_unit, backend, use_edge_dialect, dynamic):
        class AddMulModule(torch.nn.Module):
            def forward(self, a, x, b):
                y = torch.mm(a, x)
                z = torch.add(y, b)
                return z

        dynamic_shapes = None
        if dynamic:
            embedding_dim = torch.export.Dim("embedding_dim")
            dynamic_shapes = {
                "a": {1: embedding_dim},
                "x": {0: embedding_dim},
                "b": {},
            }

        _, coreml_model, _, _, _, _ = self.run_compare_torch(
            [(2, 2), (2, 2), (2, 2)],
            AddMulModule(),
            compute_unit=compute_unit,
            backend=backend,
            frontend=TorchFrontend.EXIR,
            use_edge_dialect=use_edge_dialect,
            torch_export_dynamic_shapes=dynamic_shapes,
        )

        mil_program = coreml_model._mil_program
        matmul_or_add = {}
        for op_type in ("matmul", "add"):
            matmul_or_add[op_type] = mil_program.functions["main"].find_ops(op_type=op_type)[0]

        stack_traces = {
            k: v.scopes[ScopeSource.EXIR_STACK_TRACE][0] for k, v in matmul_or_add.items()
        }
        source_codes = {
            "matmul": "y = torch.mm(a, x)",
            "add": "z = torch.add(y, b)",
        }
        for op_type in ("matmul", "add"):
            stack_trace = stack_traces[op_type]
            source_code = source_codes[op_type]
            assert stack_trace.split("\n")[-2].strip() == source_code

        if use_edge_dialect:
            debug_handle = {
                k: v.scopes[ScopeSource.EXIR_DEBUG_HANDLE][0] for k, v in matmul_or_add.items()
            }
            for v in debug_handle.values():
                assert isinstance(v, int)

            debug_handle_to_ops_mapping = mil_program.construct_debug_handle_to_ops_mapping()
            assert debug_handle_to_ops_mapping.keys() == set(debug_handle.values())

            ops = {}
            for op_type in ("matmul", "add"):
                ops[op_type] = debug_handle_to_ops_mapping[debug_handle[op_type]]
            index = {"matmul": 0, "add": 0}
            indices_const = {"matmul": (), "add": ()}
            indices_cast = {"matmul": (), "add": ()}
            if backend[1] == "fp32":
                assert len(ops["matmul"]) == 3 and len(ops["add"]) == 1
                index = {"matmul": 2, "add": 0}
                indices_const["matmul"] = (0, 1)
            else:
                # fp16 introduces additional io casts
                # each cast introduces 1 const to store destination dtype
                assert len(ops["matmul"]) == 7 and len(ops["add"]) == 5
                index = {"matmul": 6, "add": 2}
                indices_const = {"matmul": (0, 1, 2, 3), "add": (0, 3)}
                indices_cast = {"matmul": (4, 5), "add": (1, 4)}
            for op_type in ("matmul", "add"):
                assert ops[op_type][index[op_type]] == [
                    {"Type": "Program"},
                    {"Type": "Function", "Name": "main"},
                    {"Type": "Block"},
                    {
                        "Type": "Operation",
                        "Operator": op_type,
                        "Output": matmul_or_add[op_type].outputs[0].name,
                    },
                ]
                for index_const_cast in indices_const[op_type] + indices_cast[op_type]:
                    assert ops[op_type][index_const_cast][:-1] == [
                        {"Type": "Program"},
                        {"Type": "Function", "Name": "main"},
                        {"Type": "Block"},
                    ]
                for index_const in indices_const[op_type]:
                    assert ops[op_type][index_const][-1]["Operator"] == "const"
                for index_cast in indices_cast[op_type]:
                    assert ops[op_type][index_cast][-1]["Operator"] == "cast"

    @pytest.mark.parametrize(
        "compute_unit, backend, use_edge_dialect, dynamic",
        itertools.product(compute_units, backends, USE_EDGE_DIALECT, (True, False)),
    )
    def test_softmax(self, compute_unit, backend, use_edge_dialect, dynamic):
        class SoftmaxModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.softmax = torch.nn.Softmax()

            def forward(self, x):
                return self.softmax(x)

        dynamic_shapes = None
        if dynamic:
            vocab_dim = torch.export.Dim("vocab_dim")
            dynamic_shapes = {"x": {0: vocab_dim}}

        _, coreml_model, _, _, _, _ = self.run_compare_torch(
            [(2, 2)],
            SoftmaxModule(),
            compute_unit=compute_unit,
            backend=backend,
            frontend=TorchFrontend.EXIR,
            use_edge_dialect=use_edge_dialect,
            torch_export_dynamic_shapes=dynamic_shapes,
        )

        mil_program = coreml_model._mil_program
        softmax = mil_program.functions["main"].find_ops(op_type="softmax")[0]

        stack_trace = softmax.scopes[ScopeSource.EXIR_STACK_TRACE][0]
        assert stack_trace.split("\n")[-2].strip() == "return self.softmax(x)"

        if use_edge_dialect:
            debug_handle = softmax.scopes[ScopeSource.EXIR_DEBUG_HANDLE][0]
            assert isinstance(debug_handle, int)

            debug_handle_to_ops_mapping = mil_program.construct_debug_handle_to_ops_mapping()
            assert debug_handle_to_ops_mapping.keys() == {debug_handle}

            ops = debug_handle_to_ops_mapping[debug_handle]
            index_softmax = 0
            indices_const = ()
            indices_cast = ()
            if backend[1] == "fp32":
                assert len(ops) == 2
                index_softmax = 1
                indices_const = (0,)
            else:
                # fp16 introduces additional io casts
                # each cast introduces 1 const to store destination dtype
                assert len(ops) == 6
                index_softmax = 3
                indices_const = (0, 1, 4)
                indices_cast = (2, 5)
            assert ops[index_softmax] == [
                {"Type": "Program"},
                {"Type": "Function", "Name": "main"},
                {"Type": "Block"},
                {"Type": "Operation", "Operator": "softmax", "Output": softmax.outputs[0].name},
            ]
            for index_const_cast in indices_const + indices_cast:
                assert ops[index_const_cast][:-1] == [
                    {"Type": "Program"},
                    {"Type": "Function", "Name": "main"},
                    {"Type": "Block"},
                ]
            for index_const in indices_const:
                assert ops[index_const][-1]["Operator"] == "const"
            for index_cast in indices_cast:
                assert ops[index_cast][-1]["Operator"] == "cast"
