#  Copyright (c) 2023, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

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

from .testing_utils import TorchBaseTest, TorchFrontend


class TestExecutorch(TorchBaseTest):
    def test_mul(self):
        class MulModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, input, other):
                return input * other

        model = MulModule()
        model.eval()

        self.run_compare_torch(
            [(3, 2), (3, 2)],
            model,
            frontend=TorchFrontend.EDGEIR,
        )

    def test_linear(self):
        class LinearModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, 3)

            def forward(self, arg):
                return self.linear(arg)

        model = LinearModule()
        model.eval()

        self.run_compare_torch(
            [(3, 3)], model, frontend=TorchFrontend.EDGEIR, backend=("mlprogram", "fp16")
        )

    def test_add(self):
        class AddModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                z = x + y
                z = z + x
                z = z + x
                z = z + z
                return z

        model = AddModule()
        model.eval()

        self.run_compare_torch(
            [(1,), (1,)], model, frontend=TorchFrontend.EDGEIR, backend=("mlprogram", "fp16")
        )

    def test_add_mul(self):
        class AddMulModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, a, x, b):
                y = torch.mm(a, x)
                z = torch.add(y, b)
                return z

        model = AddMulModule()
        model.eval()

        self.run_compare_torch(
            [(2, 2), (2, 2), (2, 2)],
            model,
            frontend=TorchFrontend.EDGEIR,
            backend=("mlprogram", "fp16"),
        )

    def test_softmax(self):
        class LinearModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.softmax = torch.nn.Softmax()

            def forward(self, x):
                z = self.softmax(x)
                return z

        model = LinearModule()
        model.eval()

        self.run_compare_torch(
            [(2, 2)], model, frontend=TorchFrontend.EDGEIR, backend=("mlprogram", "fp16")
        )

    @pytest.mark.xfail(reason="numerical error")
    def test_deeplab_v3(self):
        model = torchvision.models.segmentation.deeplabv3_resnet50(
            weights=torchvision.models.segmentation.deeplabv3.DeepLabV3_ResNet50_Weights.DEFAULT
        )
        model.eval()

        self.run_compare_torch(
            [(1, 3, 224, 224)], model, frontend=TorchFrontend.EDGEIR, backend=("mlprogram", "fp16")
        )

    def test_edsr(self):
        model = torchsr.models.edsr_r16f64(2, True)
        model.eval()

        self.run_compare_torch(
            [(1, 3, 224, 224)], model, frontend=TorchFrontend.EDGEIR, backend=("mlprogram", "fp16")
        )

    def test_emformer_transcribe(self):
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

        model = EmformerRnntTranscriberExample()
        model.eval()

        self.run_compare_torch(
            [(1, 128, 80), (128,)], model, frontend=TorchFrontend.EDGEIR, backend=("mlprogram", "fp16")
        )

    def test_emformer_predict(self):
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

        model = EmformerRnntPredictorExample()
        model.eval()

        self.run_compare_torch(
            [torch.zeros([1, 128], dtype=int), torch.tensor([128], dtype=int)],
            model,
            input_as_shape=False,
            frontend=TorchFrontend.EDGEIR,
            backend=("mlprogram", "fp16"),
        )

    @pytest.mark.xfail(reason="numerical error")
    def test_emformer_join(self):
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

        model = EmformerRnntJoinerExample()
        model.eval()

        self.run_compare_torch(
            [(1, 128, 1024), (128,), (1, 128, 1024), (128,)],
            model,
            frontend=TorchFrontend.EDGEIR,
            backend=("mlprogram", "fp16"),
        )

    # TODO: add llama2

    def test_mobilebert(self):
        model = transformers.MobileBertModel.from_pretrained(
            "google/mobilebert-uncased", return_dict=False
        )
        model.eval()

        tokenizer = transformers.AutoTokenizer.from_pretrained("google/mobilebert-uncased")
        token = tokenizer("Hello, my dog is cute", return_tensors="pt")["input_ids"]

        self.run_compare_torch(
            token,
            model,
            input_as_shape=False,
            frontend=TorchFrontend.EDGEIR,
            backend=("mlprogram", "fp32"),
            rtol=0.005,
        )

    def test_mobilenet_v2(self):
        model = torchvision.models.mobilenet_v2(weights=torchvision.models.mobilenetv2.MobileNet_V2_Weights.DEFAULT)
        model.eval()

        self.run_compare_torch(
            [(1, 3, 224, 224)], model, frontend=TorchFrontend.EDGEIR, backend=("mlprogram", "fp16")
        )

    def test_mobilenet_v3(self):
        model = torchvision.models.mobilenet_v3_small(pretrained=True)
        model.eval()

        self.run_compare_torch(
            [(1, 3, 224, 224)], model, frontend=TorchFrontend.EDGEIR, backend=("mlprogram", "fp16")
        )

    def test_vit(self):
        model = torchvision.models.vit_b_16(weights="IMAGENET1K_V1")
        model.eval()

        self.run_compare_torch(
            [(1, 3, 224, 224)], model, frontend=TorchFrontend.EDGEIR, backend=("mlprogram", "fp16")
        )

    def test_wav2letter(self):
        model = torchaudio.models.Wav2Letter(num_classes=4096)
        model.eval()

        self.run_compare_torch(
            [(10, 1, 700)], model, frontend=TorchFrontend.EDGEIR, backend=("mlprogram", "fp16")
        )

    def test_inception_v3(self):
        model = torchvision.models.inception_v3(weights="IMAGENET1K_V1")
        model.eval()

        self.run_compare_torch(
            [(1, 3, 224, 224)], model, frontend=TorchFrontend.EDGEIR, backend=("mlprogram", "fp16")
        )

    def test_inception_v4(self):
        model = timm.models.inception_v4(pretrained=True)
        model.eval()

        self.run_compare_torch(
            [(1, 3, 299, 299)], model, frontend=TorchFrontend.EDGEIR, backend=("mlprogram", "fp16")
        )

    def test_resnet18(self):
        model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        model.eval()

        self.run_compare_torch(
            [(1, 3, 224, 224)], model, frontend=TorchFrontend.EDGEIR, backend=("mlprogram", "fp16")
        )

    def test_resnet50(self):
        model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
        model.eval()

        self.run_compare_torch(
            [(1, 3, 224, 224)], model, frontend=TorchFrontend.EDGEIR, backend=("mlprogram", "fp16")
        )
