#  Copyright (c) 2023, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import itertools
import pytest

from coremltools._deps import _HAS_EXECUTORCH, _HAS_TORCH_VISION

if not (_HAS_EXECUTORCH and _HAS_TORCH_VISION):
    pytest.skip(allow_module_level=True, reason="executorch and torchvision are required")

import torch
import torchvision
import torchaudio
import torchsr

import timm
import transformers

from coremltools.converters.mil import testing_reqs
from coremltools.converters.mil.mil.scope import ScopeSource

from .testing_utils import TorchBaseTest, TorchFrontend

backends = testing_reqs.backends
compute_units = testing_reqs.compute_units


class TestExecutorchExampleModels(TorchBaseTest):
    @pytest.mark.parametrize("compute_unit, backend", itertools.product(compute_units, backends))
    def test_mul(self, compute_unit, backend):
        class MulModule(torch.nn.Module):
            def forward(self, input, other):
                return input * other

        _, coreml_model, _, _, _, _ = self.run_compare_torch(
            [(3, 2), (3, 2)],
            MulModule(),
            compute_unit=compute_unit,
            backend=backend,
            frontend=TorchFrontend.EXIR,
        )

        mil_program = coreml_model._mil_program
        mul = mil_program.functions["main"].find_ops(op_type="mul")[0]

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

    @pytest.mark.parametrize("compute_unit, backend", itertools.product(compute_units, backends))
    def test_linear(self, compute_unit, backend):
        class LinearModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, 3)

            def forward(self, arg):
                return self.linear(arg)

        _, coreml_model, _, _, _, _ = self.run_compare_torch(
            [(3, 3)],
            LinearModule(),
            compute_unit=compute_unit,
            backend=backend,
            frontend=TorchFrontend.EXIR,
        )

        mil_program = coreml_model._mil_program
        linear = mil_program.functions["main"].find_ops(op_type="linear")[0]

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

    @pytest.mark.parametrize("compute_unit, backend", itertools.product(compute_units, backends))
    def test_add(self, compute_unit, backend):
        class AddModule(torch.nn.Module):
            def forward(self, x, y):
                z = x + y
                z = z + x
                z = z + x
                z = z + z
                return z

        _, coreml_model, _, _, _, _ = self.run_compare_torch(
            [(1,), (1,)],
            AddModule(),
            compute_unit=compute_unit,
            backend=backend,
            frontend=TorchFrontend.EXIR,
        )

        mil_program = coreml_model._mil_program
        adds = mil_program.functions["main"].find_ops(op_type="add")

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

    @pytest.mark.parametrize("compute_unit, backend", itertools.product(compute_units, backends))
    def test_add_mul(self, compute_unit, backend):
        class AddMulModule(torch.nn.Module):
            def forward(self, a, x, b):
                y = torch.mm(a, x)
                z = torch.add(y, b)
                return z

        _, coreml_model, _, _, _, _ = self.run_compare_torch(
            [(2, 2), (2, 2), (2, 2)],
            AddMulModule(),
            compute_unit=compute_unit,
            backend=backend,
            frontend=TorchFrontend.EXIR,
        )

        mil_program = coreml_model._mil_program
        matmul_or_add = {}
        for op_type in ("matmul", "add"):
            matmul_or_add[op_type] = mil_program.functions["main"].find_ops(op_type=op_type)[0]

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

    @pytest.mark.parametrize("compute_unit, backend", itertools.product(compute_units, backends))
    def test_softmax(self, compute_unit, backend):
        class SoftmaxModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.softmax = torch.nn.Softmax()

            def forward(self, x):
                return self.softmax(x)

        _, coreml_model, _, _, _, _ = self.run_compare_torch(
            [(2, 2)],
            SoftmaxModule(),
            compute_unit=compute_unit,
            backend=backend,
            frontend=TorchFrontend.EXIR,
        )

        mil_program = coreml_model._mil_program
        softmax = mil_program.functions["main"].find_ops(op_type="softmax")[0]

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

    @pytest.mark.xfail(reason="numerical error")
    @pytest.mark.parametrize("compute_unit, backend", itertools.product(compute_units, backends))
    def test_deeplab_v3(self, compute_unit, backend):
        self.run_compare_torch(
            [(1, 3, 224, 224)],
            torchvision.models.segmentation.deeplabv3_resnet50(
                weights=torchvision.models.segmentation.deeplabv3.DeepLabV3_ResNet50_Weights.DEFAULT
            ),
            compute_unit=compute_unit,
            backend=backend,
            frontend=TorchFrontend.EXIR,
        )

    @pytest.mark.parametrize("compute_unit, backend", itertools.product(compute_units, backends))
    def test_edsr(self, compute_unit, backend):
        self.run_compare_torch(
            [(1, 3, 224, 224)],
            torchsr.models.edsr_r16f64(2, True),
            compute_unit=compute_unit,
            backend=backend,
            frontend=TorchFrontend.EXIR,
        )

    @pytest.mark.parametrize("compute_unit, backend", itertools.product(compute_units, backends))
    def test_emformer_transcribe(self, compute_unit, backend):
        class EmformerRnntTranscriberExample(torch.nn.Module):
            """
            This is a wrapper for validating transcriber for the Emformer RNN-T architecture.
            It does not reflect the actual usage such as beam search, but rather an example for the export workflow.
            """

            def __init__(self) -> None:
                super().__init__()
                bundle = torchaudio.pipelines.EMFORMER_RNNT_BASE_LIBRISPEECH
                decoder = bundle.get_decoder()
                self.rnnt = decoder.model

            def forward(self, sources, source_lengths):
                return self.rnnt.transcribe(sources, source_lengths)

        if backend[0] == "neuralnetwork":
            pytest.xfail("rdar://125514139 emformer transcribe fails on neuralnetwork")

        self.run_compare_torch(
            [(1, 128, 80), (128,)],
            EmformerRnntTranscriberExample(),
            compute_unit=compute_unit,
            backend=backend,
            frontend=TorchFrontend.EXIR,
        )

    @pytest.mark.parametrize("compute_unit, backend", itertools.product(compute_units, backends))
    def test_emformer_predict(self, compute_unit, backend):
        class EmformerRnntPredictorExample(torch.nn.Module):
            """
            This is a wrapper for validating predictor for the Emformer RNN-T architecture.
            It does not reflect the actual usage such as beam search, but rather an example for the export workflow.
            """

            def __init__(self) -> None:
                super().__init__()
                bundle = torchaudio.pipelines.EMFORMER_RNNT_BASE_LIBRISPEECH
                decoder = bundle.get_decoder()
                self.rnnt = decoder.model

            def forward(self, targets, target_lengths):
                return self.rnnt.predict(targets, target_lengths, None)

        self.run_compare_torch(
            [torch.zeros([1, 128], dtype=int), torch.tensor([128], dtype=int)],
            EmformerRnntPredictorExample(),
            input_as_shape=False,
            compute_unit=compute_unit,
            backend=backend,
            frontend=TorchFrontend.EXIR,
        )

    @pytest.mark.xfail(reason="numerical error")
    @pytest.mark.parametrize("compute_unit, backend", itertools.product(compute_units, backends))
    def test_emformer_join(self, compute_unit, backend):
        class EmformerRnntJoinerExample(torch.nn.Module):
            """
            This is a wrapper for validating joiner for the Emformer RNN-T architecture.
            It does not reflect the actual usage such as beam search, but rather an example for the export workflow.
            """

            def __init__(self) -> None:
                super().__init__()
                bundle = torchaudio.pipelines.EMFORMER_RNNT_BASE_LIBRISPEECH
                decoder = bundle.get_decoder()
                self.rnnt = decoder.model

            def forward(self, source_encodings, source_lengths, target_encodings, target_lengths):
                return self.rnnt.join(source_encodings, source_lengths, target_encodings, target_lengths)

        self.run_compare_torch(
            [(1, 128, 1024), (128,), (1, 128, 1024), (128,)],
            EmformerRnntJoinerExample(),
            compute_unit=compute_unit,
            backend=backend,
            frontend=TorchFrontend.EXIR,
        )

    @pytest.mark.parametrize("compute_unit, backend", itertools.product(compute_units, backends))
    def test_mobilebert(self, compute_unit, backend):
        if backend[1] == "fp16":
            pytest.skip("Mobile Bert overflows fp16")

        tokenizer = transformers.AutoTokenizer.from_pretrained("google/mobilebert-uncased")
        token = tokenizer("Hello, my dog is cute", return_tensors="pt")["input_ids"]

        self.run_compare_torch(
            token,
            transformers.MobileBertModel.from_pretrained(
                "google/mobilebert-uncased", return_dict=False
            ),
            input_as_shape=False,
            compute_unit=compute_unit,
            backend=backend,
            frontend=TorchFrontend.EXIR,
            rtol=0.005,
        )

    @pytest.mark.parametrize("compute_unit, backend", itertools.product(compute_units, backends))
    def test_mobilenet_v2(self, compute_unit, backend):
        self.run_compare_torch(
            [(1, 3, 224, 224)],
            torchvision.models.mobilenet_v2(
                weights=torchvision.models.mobilenetv2.MobileNet_V2_Weights.DEFAULT
            ),
            compute_unit=compute_unit,
            backend=backend,
            frontend=TorchFrontend.EXIR,
        )

    @pytest.mark.parametrize("compute_unit, backend", itertools.product(compute_units, backends))
    def test_mobilenet_v3(self, compute_unit, backend):
        self.run_compare_torch(
            [(1, 3, 224, 224)],
            torchvision.models.mobilenet_v3_small(pretrained=True),
            compute_unit=compute_unit,
            backend=backend,
            frontend=TorchFrontend.EXIR,
        )

    @pytest.mark.parametrize("compute_unit, backend", itertools.product(compute_units, backends))
    def test_vit(self, compute_unit, backend):
        self.run_compare_torch(
            [(1, 3, 224, 224)],
            torchvision.models.vit_b_16(weights="IMAGENET1K_V1"),
            compute_unit=compute_unit,
            backend=backend,
            frontend=TorchFrontend.EXIR,
        )

    @pytest.mark.parametrize("compute_unit, backend", itertools.product(compute_units, backends))
    def test_wav2letter(self, compute_unit, backend):
        self.run_compare_torch(
            [(10, 1, 700)],
            torchaudio.models.Wav2Letter(num_classes=4096),
            compute_unit=compute_unit,
            backend=backend,
            frontend=TorchFrontend.EXIR,
        )

    @pytest.mark.parametrize("compute_unit, backend", itertools.product(compute_units, backends))
    def test_inception_v3(self, compute_unit, backend):
        self.run_compare_torch(
            [(1, 3, 224, 224)],
            torchvision.models.inception_v3(weights="IMAGENET1K_V1"),
            compute_unit=compute_unit,
            backend=backend,
            frontend=TorchFrontend.EXIR,
        )

    @pytest.mark.parametrize("compute_unit, backend", itertools.product(compute_units, backends))
    def test_inception_v4(self, compute_unit, backend):
        self.run_compare_torch(
            [(1, 3, 299, 299)],
            timm.models.inception_v4(pretrained=True),
            compute_unit=compute_unit,
            backend=backend,
            frontend=TorchFrontend.EXIR,
        )

    @pytest.mark.parametrize("compute_unit, backend", itertools.product(compute_units, backends))
    def test_resnet18(self, compute_unit, backend):
        self.run_compare_torch(
            [(1, 3, 224, 224)],
            torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1),
            compute_unit=compute_unit,
            backend=backend,
            frontend=TorchFrontend.EXIR,
        )

    @pytest.mark.parametrize("compute_unit, backend", itertools.product(compute_units, backends))
    def test_resnet50(self, compute_unit, backend):
        self.run_compare_torch(
            [(1, 3, 224, 224)],
            torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1),
            compute_unit=compute_unit,
            backend=backend,
            frontend=TorchFrontend.EXIR,
        )
