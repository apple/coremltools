#  Copyright (c) 2023, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import itertools
from sys import version_info

import numpy as np
import pytest

from ..utils import TorchFrontend
from .testing_utils import frontends as _frontends

frontends = _frontends.copy()

if TorchFrontend.TORCHSCRIPT in frontends:
    frontends.remove(TorchFrontend.TORCHSCRIPT)

if len(frontends) == 0:
    pytest.skip(allow_module_level=True)

from coremltools.converters.mil.frontend.torch.exir_utils import WRAPPED_SCALAR_INPUT_SUFFIX

import torch

if TorchFrontend.EXECUTORCH in frontends:
    import executorch.exir

import coremltools as ct
from coremltools.converters.mil import testing_reqs
from coremltools.converters.mil.mil.scope import ScopeSource
from coremltools.converters.mil.testing_utils import assert_spec_input_image_type, verify_prediction
from coremltools.models import _METADATA_SOURCE_DIALECT
from coremltools.proto import FeatureTypes_pb2 as ft

from .testing_utils import TorchBaseTest, export_torch_model_to_frontend

backends = testing_reqs.backends
compute_units = testing_reqs.compute_units

TORCH_EXPORT_DEFAULT_LOWER_BOUND = {TorchFrontend.TORCHEXPORT: 2, TorchFrontend.EXECUTORCH: 2}
if torch.__version__ >= "2.4.0":
    TORCH_EXPORT_DEFAULT_LOWER_BOUND[TorchFrontend.TORCHEXPORT] = 0


class TestTorchExportConversionAPI(TorchBaseTest):
    @pytest.mark.parametrize("frontend", frontends)
    def test_image_input(self, frontend):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv2d = torch.nn.Conv2d(in_channels=3, out_channels=5, kernel_size=3)

            def forward(self, x):
                return self.conv2d(x)

        model = Model()
        model.eval()

        shape = (1, 3, 128, 256)
        x = torch.randint(0, 256, shape, dtype=torch.float32)
        exported_model = export_torch_model_to_frontend(model, (x,), frontend)

        mlmodel = ct.convert(
            exported_model,
            inputs=[ct.ImageType(shape=shape, color_layout=ct.colorlayout.RGB)],
            outputs=[ct.TensorType(dtype=np.float32)],
        )

        assert_spec_input_image_type(mlmodel._spec, expected_feature_type=ft.ImageFeatureType.RGB)
        verify_prediction(mlmodel)

    @pytest.mark.parametrize("frontend", frontends)
    def test_unsupported_dim_order(self, frontend):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv3d = torch.nn.Conv3d(in_channels=3, out_channels=5, kernel_size=3)

            def forward(self, x):
                return self.conv3d(x)

        model = Model()
        model.eval()

        shape = (1, 3, 128, 256, 512)
        x = torch.randint(0, 256, shape, dtype=torch.float32)
        x = x.to(memory_format=torch.channels_last_3d)
        exported_model = export_torch_model_to_frontend(model, (x,), frontend)

        with pytest.raises(
            NotImplementedError,
            match=r"Core ML does not have general support for non-contiguous dim order",
        ):
            ct.convert(exported_model)

    @pytest.mark.parametrize(
        "compute_unit, backend, frontend",
        itertools.product(compute_units, backends, frontends),
    )
    def test_scalar_input(self, compute_unit, backend, frontend):
        class Model(torch.nn.Module):
            def forward(self, x):
                return x + 1

        model = Model()
        model.eval()

        mlmodel = self.run_compare_torch(
            (),
            model,
            compute_unit=compute_unit,
            backend=backend,
            frontend=frontend,
        )[1]
        main_function = mlmodel._mil_program.functions["main"]

        assert len(main_function.inputs) == 1
        input_name = list(main_function.inputs.keys())[0]
        input_var = main_function.inputs[input_name]
        assert input_name.endswith(WRAPPED_SCALAR_INPUT_SUFFIX)
        assert input_var.shape == (1,)

        squeeze_op = main_function.find_ops(op_type="squeeze")[0]
        if backend[1] == "fp32":
            assert squeeze_op.x is input_var
        elif backend[1] == "fp16":
            cast_op = main_function.find_ops(op_type="cast")[0]
            assert cast_op.x is input_var
            assert cast_op.dtype.val == "fp16"
            assert squeeze_op.x is cast_op.outputs[0]

    @pytest.mark.parametrize(
        "compute_unit, backend, frontend",
        itertools.product(compute_units, backends, frontends),
    )
    def test_dynamic_input(self, compute_unit, backend, frontend):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, 5)

            def forward(self, x):
                return self.linear(x)

        model = Model()
        model.eval()

        batch_dim = torch.export.Dim("batch_dim")
        dynamic_shapes = {"x": {0: batch_dim}}

        # default dynamic shape: inherit from torch.export
        coreml_model = self.run_compare_torch(
            (2, 3),
            model,
            compute_unit=compute_unit,
            backend=backend,
            frontend=frontend,
            torch_export_dynamic_shapes=dynamic_shapes,
        )[1]
        input_proto = coreml_model.input_description._fd_spec[0]
        size_ranges = input_proto.type.multiArrayType.shapeRange.sizeRanges
        assert size_ranges[0].lowerBound == TORCH_EXPORT_DEFAULT_LOWER_BOUND[frontend]
        assert size_ranges[0].upperBound == 2147483647
        assert size_ranges[1].lowerBound == 3
        assert size_ranges[1].upperBound == 3

        # customize: replace torch.export dynamic shape with Core ML enumerated shape
        shape0 = (2, 3)
        shape1 = (4, 3)
        shape2 = (8, 3)
        enumerated_shapes = ct.EnumeratedShapes(shapes=[shape0, shape1, shape2])
        coreml_model = self.run_compare_torch(
            shape1,
            model,
            compute_unit=compute_unit,
            backend=backend,
            frontend=frontend,
            torch_export_dynamic_shapes=dynamic_shapes,
            converter_input_type=[ct.TensorType(shape=enumerated_shapes)],
        )[1]
        input_proto = coreml_model.input_description._fd_spec[0]
        enumerated_shapes = input_proto.type.multiArrayType.enumeratedShapes.shapes
        assert tuple(enumerated_shapes[0].shape) == shape0
        assert tuple(enumerated_shapes[1].shape) == shape1
        assert tuple(enumerated_shapes[2].shape) == shape2

        # customize: materialize torch.export dynamic shape into static shape
        shape = (8, 3)
        coreml_model = self.run_compare_torch(
            shape,
            model,
            compute_unit=compute_unit,
            backend=backend,
            frontend=frontend,
            torch_export_dynamic_shapes=dynamic_shapes,
            converter_input_type=[ct.TensorType(shape=shape)],
        )[1]
        input_proto = coreml_model.input_description._fd_spec[0]
        proto_shape = input_proto.type.multiArrayType.shape
        assert tuple(proto_shape) == shape

    @pytest.mark.parametrize("frontend, dynamic", itertools.product(frontends, (True, False)))
    def test_invalid_inputs(self, frontend, dynamic):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, 5)

            def forward(self, x):
                return self.linear(x)

        model = Model()
        model.eval()

        example_inputs = (torch.rand(2, 3),)

        dynamic_shapes = None
        if dynamic:
            batch_dim = torch.export.Dim("batch_dim")
            dynamic_shapes = {"x": {0: batch_dim}}

        exported_program = torch.export.export(
            model,
            example_inputs,
            dynamic_shapes=dynamic_shapes,
        )
        if frontend == TorchFrontend.EXECUTORCH:
            exported_program = executorch.exir.to_edge(exported_program).exported_program()

        with pytest.raises(
            ValueError,
            match=(
                r"inconsistent shape between EXIR input x shape \((is)?[0-9]+, 3\) "
                r"and user specified input None shape \(2, 4\)"
            ),
        ):
            ct.convert(exported_program, inputs=[ct.TensorType(shape=(2, 4))])

    @staticmethod
    @pytest.mark.parametrize("backend, frontend", itertools.product(backends, frontends))
    def test_fp16_io_on_fp32_model(backend, frontend):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(10, 20)

            def forward(self, x):
                return self.linear(x)

        model = Model()
        model.eval()

        shape = (1, 10)
        example_inputs = (torch.rand(*shape),)
        exported_model = export_torch_model_to_frontend(model, example_inputs, frontend)

        # Default deployment target is iOS14 for neuralnetwork and iOS15 for mlprogram,
        # both are too old to support fp16 io
        with pytest.raises(
            TypeError,
            match=(
                r"float16 dtype for inputs is only supported for deployment target "
                r">= iOS16/macOS13/watchOS9/tvOS16"
            ),
        ):
            ct.convert(
                exported_model, convert_to=backend[0], inputs=[ct.TensorType(dtype=np.float16)]
            )

        # fp16 io should work fine for iOS16+
        if backend[0] == "mlprogram":
            ct.convert(
                exported_model,
                convert_to="mlprogram",
                minimum_deployment_target=ct.target.iOS16,
            )

    @staticmethod
    @pytest.mark.parametrize("backend, frontend", itertools.product(backends, frontends))
    def test_fp16_model(backend, frontend):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(10, 20, dtype=torch.float16)

            def forward(self, x):
                return self.linear(x)

        model = Model()
        model.eval()

        shape = (2, 10)
        example_inputs = (torch.rand(*shape, dtype=torch.float16),)
        exported_model = export_torch_model_to_frontend(model, example_inputs, frontend)

        # fp32 io should always work fine
        ct.convert(exported_model, convert_to=backend[0], inputs=[ct.TensorType(dtype=np.float32)])

        # Default deployment target is iOS14 for neuralnetwork and iOS15 for mlprogram,
        # both are too old to support fp16 io
        with pytest.raises(
            ValueError, match=r"To use fp16 input, please set minimum deployment target to iOS16\+"
        ):
            ct.convert(exported_model, convert_to=backend[0])

        # fp16 io should work fine for iOS16+
        if backend[0] == "mlprogram":
            ct.convert(
                exported_model,
                convert_to="mlprogram",
                minimum_deployment_target=ct.target.iOS16,
            )

    @pytest.mark.parametrize("backend, frontend", itertools.product(backends, frontends))
    def test_source_dialect_metadata(self, backend, frontend):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(10, 20)

            def forward(self, x):
                return self.linear(x)

        model = Model()
        model.eval()

        mlmodel = self.run_compare_torch(
            (4, 10),
            model,
            backend=backend,
            frontend=frontend,
        )[1]

        assert _METADATA_SOURCE_DIALECT in mlmodel.user_defined_metadata
        dialect_name = (
            "TorchExport::ATEN" if frontend == TorchFrontend.TORCHEXPORT else "TorchExport::EDGE"
        )
        assert mlmodel.user_defined_metadata[_METADATA_SOURCE_DIALECT] == dialect_name


class TestExecuTorchExamples(TorchBaseTest):
    @pytest.mark.parametrize(
        "compute_unit, backend, frontend, dynamic",
        itertools.product(compute_units, backends, frontends, (True, False)),
    )
    def test_mul(self, compute_unit, backend, frontend, dynamic):
        if (version_info.major, version_info.minor) == (3, 12):
            pytest.skip("rdar://139103000 (Two Unit Tests Fail only on Lighting (not locally) and with Python 3.12)")

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

        coreml_model = self.run_compare_torch(
            [(3, 2), (3, 2)],
            MulModule(),
            compute_unit=compute_unit,
            backend=backend,
            frontend=frontend,
            torch_export_dynamic_shapes=dynamic_shapes,
        )[1]

        if dynamic:
            for input_proto in coreml_model.input_description._fd_spec:
                size_ranges = input_proto.type.multiArrayType.shapeRange.sizeRanges
                assert size_ranges[0].lowerBound == TORCH_EXPORT_DEFAULT_LOWER_BOUND[frontend]
                assert size_ranges[0].upperBound == 2147483647
                assert size_ranges[1].lowerBound == max(
                    1, TORCH_EXPORT_DEFAULT_LOWER_BOUND[frontend]
                )
                assert size_ranges[1].upperBound == 3

        mil_program = coreml_model._mil_program
        mul = mil_program.functions["main"].find_ops(op_type="mul")[0]

        stack_trace = mul.scopes[ScopeSource.EXIR_STACK_TRACE][0]
        assert stack_trace.split("\n")[-2].strip() == "return input * other"

        if frontend == TorchFrontend.EXECUTORCH:
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
        "compute_unit, backend, frontend, dynamic",
        itertools.product(compute_units, backends, frontends, (True, False)),
    )
    def test_linear(self, compute_unit, backend, frontend, dynamic):
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

        coreml_model = self.run_compare_torch(
            [(3, 3)],
            LinearModule(),
            compute_unit=compute_unit,
            backend=backend,
            frontend=frontend,
            torch_export_dynamic_shapes=dynamic_shapes,
        )[1]

        if dynamic:
            input_proto = coreml_model.input_description._fd_spec[0]
            size_ranges = input_proto.type.multiArrayType.shapeRange.sizeRanges
            assert size_ranges[0].lowerBound == TORCH_EXPORT_DEFAULT_LOWER_BOUND[frontend]
            assert size_ranges[0].upperBound == 2147483647
            assert size_ranges[1].lowerBound == 3
            assert size_ranges[1].upperBound == 3

        mil_program = coreml_model._mil_program
        linear = mil_program.functions["main"].find_ops(op_type="linear")[0]

        stack_trace = linear.scopes[ScopeSource.EXIR_STACK_TRACE][0]
        assert stack_trace.split("\n")[-2].strip() == "return self.linear(arg)"

        if frontend == TorchFrontend.EXECUTORCH:
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
        "compute_unit, backend, frontend, dynamic",
        itertools.product(compute_units, backends, frontends, (True, False)),
    )
    def test_add(self, compute_unit, backend, frontend, dynamic):
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

        coreml_model = self.run_compare_torch(
            [(1,), (1,)],
            AddModule(),
            compute_unit=compute_unit,
            backend=backend,
            frontend=frontend,
            torch_export_dynamic_shapes=dynamic_shapes,
        )[1]

        if dynamic:
            for input_proto in coreml_model.input_description._fd_spec:
                size_ranges = input_proto.type.multiArrayType.shapeRange.sizeRanges
                assert size_ranges[0].lowerBound == 1
                assert size_ranges[0].upperBound == 2147483647

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

        if frontend == TorchFrontend.EXECUTORCH:
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
        "compute_unit, backend, frontend, dynamic",
        itertools.product(compute_units, backends, frontends, (True, False)),
    )
    def test_add_mul(self, compute_unit, backend, frontend, dynamic):
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

        coreml_model = self.run_compare_torch(
            [(2, 2), (2, 2), (2, 2)],
            AddMulModule(),
            compute_unit=compute_unit,
            backend=backend,
            frontend=frontend,
            torch_export_dynamic_shapes=dynamic_shapes,
        )[1]

        if dynamic:
            for i, input_proto in enumerate(coreml_model.input_description._fd_spec):
                multi_array_type = input_proto.type.multiArrayType
                shape = multi_array_type.shape
                size_ranges = multi_array_type.shapeRange.sizeRanges
                if i == 0:
                    assert size_ranges[0].lowerBound == 2
                    assert size_ranges[0].upperBound == 2
                    assert size_ranges[1].lowerBound == TORCH_EXPORT_DEFAULT_LOWER_BOUND[frontend]
                    assert size_ranges[1].upperBound == 2147483647
                elif i == 1:
                    assert size_ranges[0].lowerBound == TORCH_EXPORT_DEFAULT_LOWER_BOUND[frontend]
                    assert size_ranges[0].upperBound == 2147483647
                    assert size_ranges[1].lowerBound == 2
                    assert size_ranges[1].upperBound == 2
                else:
                    assert i == 2
                    assert shape == [2, 2]
                    assert len(size_ranges) == 0

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

        if frontend == TorchFrontend.EXECUTORCH:
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
        "compute_unit, backend, frontend, dynamic",
        itertools.product(compute_units, backends, frontends, (True, False)),
    )
    def test_softmax(self, compute_unit, backend, frontend, dynamic):
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

        coreml_model = self.run_compare_torch(
            [(2, 2)],
            SoftmaxModule(),
            compute_unit=compute_unit,
            backend=backend,
            frontend=frontend,
            torch_export_dynamic_shapes=dynamic_shapes,
        )[1]

        if dynamic:
            input_proto = coreml_model.input_description._fd_spec[0]
            size_ranges = input_proto.type.multiArrayType.shapeRange.sizeRanges
            assert size_ranges[0].lowerBound == TORCH_EXPORT_DEFAULT_LOWER_BOUND[frontend]
            assert size_ranges[0].upperBound == 2147483647
            assert size_ranges[1].lowerBound == 2
            assert size_ranges[1].upperBound == 2

        mil_program = coreml_model._mil_program
        softmax = mil_program.functions["main"].find_ops(op_type="softmax")[0]

        stack_trace = softmax.scopes[ScopeSource.EXIR_STACK_TRACE][0]
        assert stack_trace.split("\n")[-2].strip() == "return self.softmax(x)"

        if frontend == TorchFrontend.EXECUTORCH:
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
