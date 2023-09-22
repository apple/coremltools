#  Copyright (c) 2022, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import itertools
import os

import numpy as np
import pytest
from PIL import Image

import coremltools as ct
from coremltools._deps import _HAS_TORCH, MSG_TORCH_NOT_FOUND
from coremltools.converters.mil.frontend.torch.test.testing_utils import _copy_input_data
from coremltools.converters.mil.testing_reqs import backends
from coremltools.converters.mil.testing_utils import (
    assert_cast_ops_count,
    assert_input_dtype,
    assert_ops_in_mil_program,
    assert_output_dtype,
    assert_prog_input_type,
    assert_prog_output_type,
    assert_spec_input_image_type,
    assert_spec_output_image_type,
    get_op_types_in_program,
    verify_prediction,
)
from coremltools.proto import FeatureTypes_pb2 as ft
from coremltools.test.api.test_api_examples import TestInputs as _TestInputs

if _HAS_TORCH:
    import torch
    import torchvision

    torch.manual_seed(1818)

#################################################################################
# Note: all tests are also used as examples in https://coremltools.readme.io/docs
# as a reference.
# Whenever any of the following test fails, we should update API documentations
#################################################################################

@pytest.mark.skipif(not _HAS_TORCH, reason=MSG_TORCH_NOT_FOUND)
class TestPyTorchConverterExamples:
    @staticmethod
    @pytest.mark.parametrize(
        "backend",
        backends,
    )
    def test_convert_torch_vision_mobilenet_v2(tmpdir, backend):
        """
        In this example, we'll instantiate a PyTorch classification model and convert
        it to Core ML.
        """

        """
        Here we instantiate our model. In a real use case this would be your trained
        model.
        """
        model = torchvision.models.mobilenet_v2()

        """
        The next thing we need to do is generate TorchScript for the model. The easiest
        way to do this is by tracing it.
        """

        """
        It's important that a model be in evaluation mode (not training mode) when it's
        traced. This makes sure things like dropout are disabled.
        """
        model.eval()

        """
        Tracing takes an example input and traces its flow through the model. Here we
        are creating an example image input.

        The rank and shape of the tensor will depend on your model use case. If your
        model expects a fixed size input, use that size here. If it can accept a
        variety of input sizes, it's generally best to keep the example input small to
        shorten how long it takes to run a forward pass of your model. In all cases,
        the rank of the tensor must be fixed.
        """
        example_input = torch.rand(1, 3, 256, 256)

        """
        Now we actually trace the model. This will produce the TorchScript that the
        CoreML converter needs.
        """
        traced_model = torch.jit.trace(model, example_input)

        """
        Now with a TorchScript representation of the model, we can call the CoreML
        converter. The converter also needs a description of the input to the model,
        where we can give it a convenient name.
        """
        mlmodel = ct.convert(
            traced_model,
            inputs=[ct.TensorType(name="input", shape=example_input.shape)],
            convert_to=backend[0],
        )

        """
        Now with a conversion complete, we can save the MLModel and run inference.
        """
        suffix = ".mlmodel" if backend == "neuralnetwork" else ".mlpackage"
        save_path = os.path.join(str(tmpdir), "mobilenet_v2" + suffix)
        mlmodel.save(save_path)

        """
        Running predict() is only supported on macOS.
        """
        if ct.utils._is_macos():
            results = mlmodel.predict({"input": example_input.numpy()})
            assert isinstance(results, dict)

    @staticmethod
    def test_convert_torch_traced_model_to_milinternal(tmpdir):
        from torch import nn
        class Network(nn.Module):
            def __init__(self):
                super(Network, self).__init__()
                self.hidden = nn.Linear(100, 10)
                self.output = nn.Linear(10, 2)
                self.sigmoid = nn.Sigmoid()
                self.softmax = nn.Softmax(dim=1)

            def forward(self, x):
                x = self.hidden(x)
                x = self.sigmoid(x)
                x = self.output(x)
                x = self.softmax(x)
                return x

        torch_model = Network()
        torch_model.eval()
        example_input = torch.rand(1, 100)
        traced_model = torch.jit.trace(torch_model, example_input)
        model = ct.convert(
            traced_model,
            inputs=[ct.TensorType(name="input", shape=example_input.shape)],
            convert_to='milinternal'
        )
        assert isinstance(model, ct.converters.mil.Program)

    @staticmethod
    def test_torch_classifier():
        class Net(torch.nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.linear1 = torch.nn.Linear(28 * 28, 100)
                self.linear2 = torch.nn.Linear(100, 50)
                self.final = torch.nn.Linear(50, 10)
                self.relu = torch.nn.ReLU()

            def forward(self, img):  # convert + flatten
                x = img.view(-1, 28 * 28)
                x = self.relu(self.linear1(x))
                x = self.relu(self.linear2(x))
                x = self.final(x)
                return x
        model = Net()
        model.eval()
        example_input = torch.rand(1, 28 * 28, 1)
        traced_model = torch.jit.trace(model, example_input)
        traced_model.eval()

        def _test_classifier(traced_model, example_input, class_type, backend):
            label = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            if class_type == "str":
                label = list(map(lambda x: str(x), label))
            classifier_config = ct.ClassifierConfig(label)
            mlmodel = ct.convert(
                traced_model,
                source='pytorch',
                convert_to=backend,
                inputs=[
                    ct.TensorType(
                        name="input",
                        shape=example_input.shape,
                        dtype=example_input.numpy().dtype,
                    )
                ],
                classifier_config=classifier_config
            )
            if ct.utils._is_macos():
                coreml_out = mlmodel.predict({"input": example_input.detach().numpy()})
                assert "classLabel" in coreml_out
                key_type = str if class_type == "str" else int
                assert isinstance(coreml_out["classLabel"], key_type)

        for class_type in ("str", "int"):
            _test_classifier(traced_model, example_input, class_type, "neuralnetwork")
            if ct.utils._macos_version() >= (12, 0):
                _test_classifier(traced_model, example_input, class_type, "mlprogram")

    @staticmethod
    @pytest.mark.parametrize("backend", backends)
    def test_convert_to_argument_with_torch_model(tmpdir, backend):
        class Network(torch.nn.Module):
            def __init__(self):
                super(Network, self).__init__()
                self.hidden = torch.nn.Linear(30, 5)
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                x = self.hidden(x)
                return self.relu(x)

        torch_model = Network()
        torch_model.eval()
        example_input = torch.rand(1, 30)
        traced_model = torch.jit.trace(torch_model, example_input)
        model = ct.convert(
            traced_model,
            inputs=[ct.TensorType(name="input", shape=example_input.shape)],
            convert_to=backend[0],
        )
        assert isinstance(model, ct.models.MLModel)
        spec = model.get_spec()
        if backend[0] == "mlprogram":
            assert spec.WhichOneof('Type') == 'mlProgram'
        else:
            assert spec.WhichOneof('Type') == 'neuralNetwork'

    @staticmethod
    def test_deployment_target_argument_with_torch_model():
        class Network(torch.nn.Module):
            def __init__(self):
                super(Network, self).__init__()
                self.hidden = torch.nn.Linear(30, 5)
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                x = self.hidden(x)
                return self.relu(x)

        torch_model = Network()
        torch_model.eval()
        example_input = torch.rand(1, 30)
        traced_model = torch.jit.trace(torch_model, example_input)

        # convert to 'neuralnetwork' by specifying an iOS13 target
        model = ct.convert(
            traced_model,
            inputs=[ct.TensorType(name="input", shape=example_input.shape)],
            minimum_deployment_target=ct.target.iOS13,
        )
        assert isinstance(model, ct.models.MLModel)
        assert model.get_spec().WhichOneof('Type') == 'neuralNetwork'

        # convert to 'mlprogram' by specifying an iOS15 target
        model = ct.convert(
            traced_model,
            inputs=[ct.TensorType(name="input", shape=example_input.shape)],
            minimum_deployment_target=ct.target.iOS15,
        )
        assert isinstance(model, ct.models.MLModel)
        assert model.get_spec().WhichOneof('Type') == 'mlProgram'

        # verify an error is raised when convert_to="neuralnetwork" and target is iOS15
        with pytest.raises(ValueError) as e:
            model = ct.convert(
                traced_model,
                inputs=[ct.TensorType(name="input", shape=example_input.shape)],
                convert_to="neuralnetwork",
                minimum_deployment_target=ct.target.iOS15,
            )
        expected_error = "If minimum deployment target is iOS15/macOS12/watchOS8/tvOS15 or higher, " \
                         "then 'convert_to' cannot be neuralnetwork. It must be 'mlprogram'"
        assert expected_error == str(e.value)

        # verify an error is raised when convert_to="mlprogram" and target is less than iOS15
        with pytest.raises(ValueError) as e:
            model = ct.convert(
                traced_model,
                inputs=[ct.TensorType(name="input", shape=example_input.shape)],
                convert_to="mlprogram",
                minimum_deployment_target=ct.target.iOS14,
            )
        expected_error = "When 'convert_to' is mlprogram, the minimum deployment target " \
                         "must be at least iOS15/macOS12/watchOS8/tvOS15"
        assert expected_error == str(e.value)

    @staticmethod
    def test_get_milprogram_method_with_torch_model():
        class Network(torch.nn.Module):
            def __init__(self):
                super(Network, self).__init__()
                self.hidden = torch.nn.Linear(100, 10)
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                x = self.hidden(x)
                x = self.relu(x)
                return x

        torch_model = Network()
        torch_model.eval()
        example_input = torch.rand(1, 100)
        traced_model = torch.jit.trace(torch_model, example_input)
        model = ct.convert(
            traced_model,
            inputs=[ct.TensorType(shape=example_input.shape)],
            convert_to='mlprogram'
        )
        assert isinstance(model._get_mil_internal(), ct.converters.mil.Program)

    @staticmethod
    @pytest.mark.skipif(ct.utils._macos_version() < (12, 0), reason='Model produces specification 6.')
    @pytest.mark.parametrize(
        "backend, provide_prob_output_argument",
        itertools.product(
            backends,
            [False, True],
        )
    )
    def test_classifier_from_torch_model(backend, provide_prob_output_argument):
        torch_model = torch.nn.ReLU().eval()
        traced_model = torch.jit.trace(torch_model, torch.rand(3,))
        variable_name = "var_2"
        class_label_name = "class_label"
        classifier_config = ct.ClassifierConfig(
                            class_labels=['a', 'b', 'c'],
                            predicted_feature_name=class_label_name,
                            predicted_probabilities_output=variable_name if provide_prob_output_argument else None,
                            )

        model = ct.convert(
            traced_model,
            inputs=[ct.TensorType(shape=(3,))],
            classifier_config = classifier_config,
            convert_to=backend[0],
        )
        spec = model.get_spec()
        input_name = spec.description.input[0].name
        out_dict = model.predict({input_name : np.array([1.0, 2.0, 3.0])})

        assert class_label_name in out_dict
        assert out_dict[class_label_name] == 'c'
        if backend[0] == "neuralnetwork":
            assert variable_name in out_dict
            assert isinstance(out_dict[variable_name], dict)
        else:
            output_dict_feature_name = class_label_name + "_probs"
            assert output_dict_feature_name in out_dict
            assert isinstance(out_dict[output_dict_feature_name], dict)

###############################################################################
# Note: Stress tests for PyTorch input / output types
###############################################################################

@pytest.mark.skipif(ct.utils._macos_version() < (10, 15), reason='Model produces specification 4.')
@pytest.mark.skipif(not _HAS_TORCH, reason=MSG_TORCH_NOT_FOUND)
class TestTorchInputs(_TestInputs):
    @staticmethod
    @pytest.mark.skipif(not ct.utils._is_macos(), reason="test needs predictions")
    @pytest.mark.parametrize(
        "backend",
        backends,
    )
    def test_torch_predict_input(backend):
        TestTorchInputs._test_variant_input_type_prediction(torch.tensor, backend[0])

    @staticmethod
    @pytest.mark.parametrize(
        "backend",
        backends,
    )
    def test_int64_inputs(backend):

        num_tokens = 3
        embedding_size = 5

        class TestModule(torch.nn.Module):
            def __init__(self):
                super(TestModule, self).__init__()
                self.embedding = torch.nn.Embedding(num_tokens,
                    embedding_size)

            def forward(self, x):
                return self.embedding(x)

        model = TestModule()
        model.eval()

        example_input = torch.randint(high=num_tokens, size=(2,), dtype=torch.int64)
        traced_model = torch.jit.trace(model, example_input)
        mlmodel = ct.convert(
            traced_model,
            inputs=[
                ct.TensorType(
                    name="input",
                    shape=example_input.shape,
                    dtype=example_input.numpy().dtype,
                )
            ],
            convert_to=backend[0],
        )

        # running predict() is supported on macOS
        if ct.utils._is_macos():
            result = mlmodel.predict(
                {"input": example_input.detach().numpy().astype(np.float32)}
            )

            # Verify outputs
            expected = model(example_input)
            name = list(result.keys())[0]
            rtol = 1e-03 if backend[0] == "mlprogram" else 1e-07
            atol = 1e-04 if backend[0] == "mlprogram" else 0
            np.testing.assert_allclose(
                result[name], expected.detach().numpy(), rtol=rtol, atol=atol
            )

        # Duplicated inputs are invalid
        with pytest.raises(ValueError, match=r"Duplicated inputs"):
            mlmodel = ct.convert(
                traced_model,
                inputs=[
                    ct.TensorType(
                        name="input",
                        shape=example_input.shape,
                        dtype=example_input.numpy().dtype,
                    ),
                    ct.TensorType(
                        name="input",
                        shape=example_input.shape,
                        dtype=example_input.numpy().dtype,
                    ),
                ],
                convert_to=backend[0],
            )

        # Outputs must be of type ct.ImageType or ct.TensorType
        with pytest.raises(ValueError, match=r"must be a list of type ct.TensorType or ct.ImageType"):
            mlmodel = ct.convert(
                traced_model,
                inputs=[
                    ct.TensorType(
                        name="input",
                        shape=example_input.shape,
                        dtype=example_input.numpy().dtype,
                    ),
                ],
                outputs=["output"],
                convert_to=backend[0],
            )

    @staticmethod
    @pytest.mark.parametrize(
        "backend",
        backends,
    )
    def test_fully_dynamic_inputs(backend):
        """
        All dims of the inputs are dynamic, and write to slice to one of the
        inputs.
        """

        class Model(torch.nn.Module):
            def __init__(self, index):
                super(Model, self).__init__()
                self.index = index

            def forward(self, x, y):
                x[:, int(self.index.item())] = 0.0
                y = y.unsqueeze(0)
                return y, x

        model = Model(torch.tensor(3))
        scripted_model = torch.jit.script(model)

        a, b = (-1, -1) if backend[0] == "neuralnetwork" else (6, 6)

        mlmodel = ct.convert(
            scripted_model,
            inputs=[
                ct.TensorType("x", shape=(ct.RangeDim(upper_bound=a), ct.RangeDim(upper_bound=b))),
                ct.TensorType("y", shape=(ct.RangeDim(upper_bound=a), ct.RangeDim(upper_bound=b))),
            ],
            convert_to=backend[0],
        )

        # running predict() is supported on macOS
        if ct.utils._is_macos():
            x, y = torch.rand(2, 4), torch.rand(1, 2)
            torch_input = _copy_input_data([x, y])
            torch_res = model(*torch_input)
            results = mlmodel.predict({"x": x.cpu().detach().numpy(),
              "y": y.cpu().detach().numpy()})

            rtol = 1e-03 if backend[0] == "mlprogram" else 1e-07
            atol = 1e-04 if backend[0] == "mlprogram" else 0
            for i, name in enumerate(mlmodel.output_description):
                np.testing.assert_allclose(torch_res[i], results[name], rtol=rtol, atol=atol)

            x, y = torch.rand(1, 6), torch.rand(2, 3)
            torch_input = _copy_input_data([x, y])
            torch_res = model(*torch_input)
            results = mlmodel.predict({"x": x.cpu().detach().numpy(),
              "y": y.cpu().detach().numpy()})
            for i, name in enumerate(mlmodel.output_description):
                np.testing.assert_allclose(torch_res[i], results[name], rtol=rtol, atol=atol)

    @staticmethod
    @pytest.mark.parametrize(
        "backend",
        backends,
    )
    def test_rank0_inputs_torch(backend):
        """Similar to TestPyTorchConverterExamples::test_int64_inputs but
        using rank-0 int input.
        """

        num_tokens = 3
        embedding_size = 5

        class TestModule(torch.nn.Module):
            def __init__(self):
                super(TestModule, self).__init__()
                self.embedding = torch.nn.Embedding(num_tokens,
                    embedding_size)

            def forward(self, x):
                return self.embedding(x)

        model = TestModule()
        model.eval()

        example_input = torch.tensor(1)
        traced_model = torch.jit.trace(model, example_input)
        with pytest.raises(ValueError, match=r"Rank-0"):
            mlmodel = ct.convert(
                traced_model,
                inputs=[
                    ct.TensorType(
                        name="input",
                        shape=example_input.shape,
                        dtype=example_input.numpy().dtype,
                    )
                ],
                convert_to=backend[0],
            )

    @staticmethod
    @pytest.mark.parametrize(
        "variable_length, backend",
        itertools.product([True, False], backends),
    )
    def test_torch_range_dim_lstm(variable_length, backend):
        """
        This example shows how to run LSTM with previous hidden / cell states
        """

        input_size = 3
        hidden_size = 2

        class TestNet(torch.nn.Module):
            def __init__(self):
              super(TestNet, self).__init__()
              self.lstm = torch.nn.LSTM(input_size, hidden_size, 1)

            def forward(self, x, hidden_state, cell_state):
                # LSTM takes in previous hidden and cell states. The first
                # invokation usually have zero vectors as initial states.
                output, (new_hidden_state, new_cell_state) = \
                    self.lstm(x, (hidden_state, cell_state))
                # LSTM hidden / cell states are returned to be managed by the
                # caller (and is fed in as inputs in the next call).
                return output, new_hidden_state, new_cell_state

        model = TestNet()
        model.eval()

        seq_len = 2 # we'll make seq_len dynamic later
        batch = 1
        input_shape = (seq_len, batch, input_size)
        rand_input = torch.rand(*input_shape)
        h_shape = (1, batch, hidden_size)
        rand_h0 = torch.rand(*h_shape)
        rand_c0 = torch.rand(*h_shape)

        traced_model = torch.jit.trace(model, (rand_input, rand_h0, rand_c0))

        # ct.RangeDim() tells coremltools that this dimension can change for
        # each inference example (aka "runtime-determined"). If the sequence
        # length is always the same (e.g., 2 step LSTM would have seq_len == 2)
        # Note that fixed-length models usually run slightly faster than variable length models.
        upper_bound = -1 if backend[0] == "neuralnetwork" else 10
        ct_seq_len = ct.RangeDim(upper_bound=upper_bound) if variable_length else seq_len
        seq_input = ct.TensorType(shape=(ct_seq_len, batch, input_size),
            name="seq_input")
        h_input = ct.TensorType(shape=h_shape, name="h_input")
        c_input = ct.TensorType(shape=h_shape, name="c_input")

        mlmodel = ct.convert(
            traced_model,
            inputs=[seq_input, h_input, c_input],
            convert_to=backend[0],
        )

        if ct.utils._is_macos():
            result = mlmodel.predict(
                {"seq_input": rand_input.detach().numpy().astype(np.float32),
                  "h_input": rand_h0.detach().numpy().astype(np.float32),
                  "c_input": rand_c0.detach().numpy().astype(np.float32),
                  }
            )

            # Verify outputs
            expected = model(rand_input, rand_h0, rand_c0)
            names = list(result.keys())
            names.sort()
            atol = 1e-03 if backend[0] == "mlprogram" else 1e-04
            rtol = 1e-03 if backend[0] == "mlprogram" else 1e-07
            np.testing.assert_allclose(
                result[names[0]], expected[0].detach().numpy(), atol=atol, rtol=rtol
            )
            np.testing.assert_allclose(
                result[names[1]], expected[1].detach().numpy(), atol=atol, rtol=rtol
            )
            np.testing.assert_allclose(
                result[names[2]], expected[2].detach().numpy(), atol=atol, rtol=rtol
            )

            # Try example of different length
            if variable_length:
                seq_len = 10
                input_shape = (seq_len, batch, input_size)
                rand_input = torch.rand(*input_shape)

                result = mlmodel.predict(
                    {"seq_input": rand_input.detach().numpy().astype(np.float32),
                      "h_input": rand_h0.detach().numpy().astype(np.float32),
                      "c_input": rand_c0.detach().numpy().astype(np.float32),
                      }
                )
                expected = model(rand_input, rand_h0, rand_c0)
                names = list(result.keys())
                names.sort()
                np.testing.assert_allclose(
                    result[names[0]], expected[0].detach().numpy(), atol=atol, rtol=rtol
                )
                np.testing.assert_allclose(
                    result[names[1]], expected[1].detach().numpy(), atol=atol, rtol=rtol
                )
                np.testing.assert_allclose(
                    result[names[2]], expected[2].detach().numpy(), atol=atol, rtol=rtol
                )

    @staticmethod
    @pytest.mark.parametrize(
        "use_symbol, backend",
        itertools.product(
            [True, False],
            backends,
        ),
    )
    def test_torch_outofbound_range_dim(use_symbol, backend):

        num_tokens = 3
        embedding_size = 5

        class TestModule(torch.nn.Module):
            def __init__(self):
                super(TestModule, self).__init__()
                self.embedding = torch.nn.Embedding(num_tokens, embedding_size)

            def forward(self, x):
                return self.embedding(x)

        model = TestModule()
        model.eval()

        example_input = torch.randint(high=num_tokens, size=(3,),
                dtype=torch.int64)
        traced_model = torch.jit.trace(model, example_input)

        if use_symbol:
            seq_len_dim = ct.RangeDim(symbol='len', lower_bound=3,
                    upper_bound=5)
        else:
            # symbol is optional
            seq_len_dim = ct.RangeDim(lower_bound=3, upper_bound=5)
        seq_input = ct.TensorType(name="input", shape=(seq_len_dim,),
                dtype=np.int64)
        mlmodel = ct.convert(
            traced_model,
            inputs=[seq_input],
            convert_to=backend[0],
        )

        if ct.utils._is_macos():
            result = mlmodel.predict(
                {"input": example_input.detach().numpy().astype(np.float32)}
            )

            # Verify outputs
            rtol = 1e-03 if backend[0] == "mlprogram" else 1e-07
            atol = 1e-04 if backend[0] == "mlprogram" else 0
            expected = model(example_input)
            name = list(result.keys())[0]
            np.testing.assert_allclose(
                result[name], expected.detach().numpy(), rtol=rtol, atol=atol
            )

            # seq_len below/above lower_bound/upper_bound
            with pytest.raises(RuntimeError,
                    match=r"Size \(99\) of dimension \(0\) is not in allowed range \(3\.\.5\)"):
                example_input2 = torch.randint(high=num_tokens, size=(99,),
                        dtype=torch.int64)
                result = mlmodel.predict(
                    {"input": example_input2.detach().numpy().astype(np.float32)}
                )

            with pytest.raises(RuntimeError,
                    match=r"Size \(2\) of dimension \(0\) is not in allowed range \(3\.\.5\)"):
                example_input2 = torch.randint(high=num_tokens, size=(2,),
                        dtype=torch.int64)
                result = mlmodel.predict(
                    {"input": example_input2.detach().numpy().astype(np.float32)}
                )

    @staticmethod
    @pytest.mark.parametrize(
        "backend",
        backends,
    )
    def test_torch_enumerated_shapes(backend):

        in_channels = 3
        out_channels = 2
        kernel_size = 3

        class TestModule(torch.nn.Module):
            def __init__(self):
                super(TestModule, self).__init__()
                self.conv = torch.nn.Conv2d(in_channels, out_channels,
                        kernel_size)

            def forward(self, x):
                return self.conv(x)

        model = TestModule()
        model.eval()

        example_input = torch.randn(1, 3, 28, 28)
        traced_model = torch.jit.trace(model, example_input)

        shapes = [(1, 3, 28, 28), (1, 3, 56, 56)]
        enumerated_shapes = ct.EnumeratedShapes(shapes=shapes)
        tensor_input = ct.TensorType(name="input", shape=enumerated_shapes)

        mlmodel = ct.convert(
            traced_model,
            inputs=[tensor_input],
            compute_units=ct.ComputeUnit.CPU_ONLY,
            convert_to=backend[0],
        )

        if ct.utils._is_macos():
            result = mlmodel.predict(
                {"input": example_input.detach().numpy().astype(np.float32)},
            )

            # Verify outputs
            rtol = 1 if backend[0] == "mlprogram" else 1e-03
            atol = 1e-02 if backend[0] == "mlprogram" else 1e-04
            expected = model(example_input)
            name = list(result.keys())[0]
            np.testing.assert_allclose(
                result[name], expected.detach().numpy(), rtol=rtol, atol=atol
            )

            # Test (1, 3, 56, 56) shape (can't verify numerical parity with Torch
            # which doesn't support enumerated shape)
            test_input_x = np.random.rand(*shapes[1]).astype(np.float32)
            mlmodel.predict({"input": test_input_x})

            # Test with a wrong shape
            with pytest.raises(RuntimeError,
                    match=r"MultiArray Shape \(1 x 3 x 29 x 29\) was not in enumerated set of allowed shapes"):
                test_input_x = np.random.rand(1, 3, 29, 29).astype(np.float32)
                mlmodel.predict({"input": test_input_x})

    @staticmethod
    @pytest.mark.parametrize(
        "backend",
        backends,
    )
    def test_torch_image_enumerated_shapes(backend):
        import torchvision
        torch_model = torchvision.models.mobilenet_v2().features
        torch_model.eval()
        example_input = torch.rand(1, 3, 256, 256)
        traced_model = torch.jit.trace(torch_model, example_input)
        input_shapes = ct.EnumeratedShapes(shapes=[(1, 3, 256, 256), (1, 3, 224, 224)])
        image_input = ct.ImageType(shape=input_shapes,
                                   bias=[-1, -1, -1], scale=1 / 127)
        model = ct.convert(traced_model, inputs=[image_input], convert_to=backend[0])
        assert model is not None
        spec = model.get_spec()
        assert len(spec.description.input[0].type.imageType.enumeratedSizes.sizes) == 2

    @staticmethod
    @pytest.mark.parametrize(
        "backend",
        backends,
    )
    def test_torch_optional_input(backend):

        num_tokens = 3
        embedding_size = 5

        class TestModule(torch.nn.Module):
            def __init__(self):
                super(TestModule, self).__init__()
                self.embedding = torch.nn.Embedding(num_tokens, embedding_size)

            def forward(self, x, y):
                return self.embedding(x) + y

        model = TestModule()
        model.eval()

        example_input = [
            torch.randint(high=num_tokens, size=(2,), dtype=torch.int64),
            torch.rand(1),
            ]
        traced_model = torch.jit.trace(model, example_input)

        upper_bound = -1 if backend[0] == "neuralnetwork" else 2
        required_input = ct.TensorType(
            name="required_input", shape=(ct.RangeDim(upper_bound=upper_bound),), dtype=np.int64
        )
        default_value = np.array([3]).astype(np.float32)
        optional_input = ct.TensorType(name="optional_input", shape=(1,),
            default_value=default_value)

        for compute_units in ct.ComputeUnit:
            if compute_units == ct.ComputeUnit.CPU_AND_NE and ct.utils._macos_version() < (13, 0):
                continue

            mlmodel = ct.convert(
                traced_model,
                inputs=[required_input, optional_input],
                compute_units=compute_units,
                convert_to=backend[0],
            )

            assert(mlmodel.compute_unit == compute_units)

            if ct.utils._is_macos():
                result = mlmodel.predict(
                    {"required_input":
                     example_input[0].detach().numpy().astype(np.float32)}
                )

                # Verify outputs
                rtol = 1e-03 if backend[0] == "mlprogram" else 1e-07
                atol = 1e-03 if backend[0] == "mlprogram" else 0
                torch_default_value = torch.tensor([3])
                expected = model(example_input[0].detach(), torch_default_value)
                name = list(result.keys())[0]
                np.testing.assert_allclose(
                    result[name], expected.detach().numpy(), rtol=rtol, atol=atol
                )


@pytest.fixture
def int32_input_model():
    class Model(torch.nn.Module):
        def forward(self, x):
            return x + 5
    example_input = torch.randint(0, 100, (10, 20), dtype=torch.int32)
    return torch.jit.trace(Model().eval(), example_input)

def int64_input_model():
    class Model(torch.nn.Module):
        def forward(self, x):
            return x + 5
    example_input = torch.randint(0, 100, (10, 20), dtype=torch.int64)
    return torch.jit.trace(Model().eval(), example_input)

@pytest.fixture
def float32_input_model_add_op():
    class Model(torch.nn.Module):
        def forward(self, x):
            return x + 5.5
    example_input = torch.randint(0, 100, (10, 20), dtype=torch.float32)
    return torch.jit.trace(Model().eval(), example_input)

@pytest.fixture
def float32_input_model_relu_ops():
    class Model(torch.nn.Module):
        def forward(self, x):
            x = torch.nn.ReLU()(x)
            return torch.nn.ReLU()(x)
    example_input = torch.randint(0, 100, (10, 20), dtype=torch.float32)
    return torch.jit.trace(Model().eval(), example_input)

@pytest.fixture
def float32_two_input_model():
    class Model(torch.nn.Module):
        def forward(self, x, y):
            return x + y
    example_input = torch.randint(0, 100, (10, 20), dtype=torch.float32)
    return torch.jit.trace(Model().eval(), [example_input, example_input])

@pytest.fixture
def float32_two_output_model():
    class Model(torch.nn.Module):
        def forward(self, x):
            y = torch.nn.ReLU()(x)
            out1 = torch.nn.ReLU()(y)
            out2 = torch.nn.ReLU6()(x)
            return out1, out2
    example_input = torch.randint(0, 100, (10, 20), dtype=torch.float32)
    return torch.jit.trace(Model().eval(), example_input)

@pytest.fixture
def int32_float32_two_output_model():
    class Model(torch.nn.Module):
        def forward(self, x, y):
            out1 = x + 1
            out2 = y + 1
            return out1, out2

    input_1 = torch.randint(0, 100, (10, 20), dtype=torch.int32)
    input_2 = torch.randint(0, 100, (10, 20), dtype=torch.float32)
    return torch.jit.trace(Model().eval(), [input_1, input_2])


def float64_input_model():
    class Model(torch.nn.Module):
        def forward(self, x):
            return x + 5.1

    example_input = torch.randint(0, 100, (10, 20), dtype=torch.float64)
    return torch.jit.trace(Model().eval(), example_input)


@pytest.fixture
def rank3_input_model():
    class Model(torch.nn.Module):
        def forward(self, x):
            return x + 5.5
    example_input = torch.randint(0, 100, (1, 10, 20), dtype=torch.float32)
    return torch.jit.trace(Model().eval(), example_input)

@pytest.fixture
def rank4_input_model():
    class Model(torch.nn.Module):
        def forward(self, x):
            return x + 5.5
    example_input = torch.randint(0, 100, (1, 3, 10, 20), dtype=torch.float32)
    return torch.jit.trace(Model().eval(), example_input)

@pytest.fixture
def rank4_grayscale_input_model():
    class Model(torch.nn.Module):
        def forward(self, x):
            return x + 10
    example_input = torch.randint(0, 100, (1, 1, 10, 20), dtype=torch.float32)
    return torch.jit.trace(Model().eval(), example_input)

@pytest.fixture
def linear_model():
    # this model will test the fuse_linear_bias pass
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 15, bias=False)
            self.constant_tensor = torch.ones((15,), dtype=torch.float32)

        def forward(self, x):
            x = self.linear(x)
            x = x - self.constant_tensor
            x = torch.nn.ReLU()(x)
            return x
    example_input = torch.randint(0, 10, (1, 10), dtype=torch.float32)
    return torch.jit.trace(Model().eval(), example_input)


@pytest.mark.skipif(ct.utils._macos_version() < (13, 0), reason='Tests are for deployment target ios16/macos13')
class TestInputOutputConversionAPI:

    def test_input_dtype_default(self, int32_input_model):
        #if dtype is not provided it defaults to float32
        mlmodel = ct.convert(int32_input_model,
                             inputs=[ct.TensorType(shape=(10, 20))],
                             minimum_deployment_target=ct.target.macOS12)
        assert_input_dtype(mlmodel, expected_type_str="fp32")
        verify_prediction(mlmodel)

    def test_input_shape_missing_error(self, float32_input_model_add_op):
        with pytest.raises(ValueError,
                           match="'shape' must be provided in the 'inputs' argument for pytorch conversion"):
            mlmodel = ct.convert(float32_input_model_add_op,
                                 inputs=[ct.TensorType(dtype=np.int32)],
                                 minimum_deployment_target=ct.target.macOS12)

    @pytest.mark.parametrize(
        "default_input_dtype, model",
        itertools.product(
            [True, False],
            [int64_input_model, float64_input_model],
        ),
    )
    def test_unsupported_input_dtype_torch_model(self, default_input_dtype, model):
        # test that no error is raised when the Torch model's input dtype is not supported.
        # If users don't provide the input type, it will be mapped to the default dtype which is float32.
        # If the input type is provided, it will be mapped to the most compatible dtype:
        # fp64 -> fp32, int64 -> int32
        if default_input_dtype:
            dtype = None
            expected_type_str = "fp32"
        else:
            if model == int64_input_model:
                dtype = np.int64
                expected_type_str = "int32"
            elif model == float64_input_model:
                dtype = np.float64
                expected_type_str = "fp32"

        mlmodel = ct.convert(
            model(),
            inputs=[ct.TensorType(shape=(10, 20), dtype=dtype)],
            minimum_deployment_target=ct.target.macOS12,
        )
        assert_input_dtype(mlmodel, expected_type_str=expected_type_str)
        verify_prediction(mlmodel)

    def test_input_dtype_user_provided(self, float32_input_model_add_op):
        # test that provided dtype in the api is applied
        mlmodel = ct.convert(float32_input_model_add_op,
                             inputs=[ct.TensorType(shape=(10, 20), dtype=np.int32)],
                             minimum_deployment_target=ct.target.macOS12)
        assert_input_dtype(mlmodel, expected_type_str="int32")
        assert_output_dtype(mlmodel, expected_type_str="fp32")
        verify_prediction(mlmodel)

    def test_invalid_input_dtype(self, int32_input_model):
        with pytest.raises(TypeError,
                           match="is unsupported for inputs/outputs of the model"
                           ):
            mlmodel = ct.convert(int32_input_model,
                                 inputs=[ct.TensorType(dtype=np.int16)],
                                 minimum_deployment_target=ct.target.macOS12)

        with pytest.raises(TypeError,
                           match="float16 dtype for inputs is only supported for deployment target >= iOS16/macOS13"
                           ):
            mlmodel = ct.convert(int32_input_model,
                                 inputs=[ct.TensorType(dtype=np.float16)],
                                 minimum_deployment_target=ct.target.macOS12)

    def test_fp16_input_dtype(self, float32_input_model_add_op, float32_input_model_relu_ops, int32_input_model):
        """
        Test that providing fp16 input dtype works with macOS13.
        """
        mlmodel = ct.convert(
            float32_input_model_add_op,
            inputs=[ct.TensorType(shape=(10, 20), dtype=np.float16)],
            outputs=[ct.TensorType(dtype=np.float32)],
            minimum_deployment_target=ct.target.macOS13,
        )
        assert_ops_in_mil_program(mlmodel, expected_op_list=["add", "cast"])
        assert_input_dtype(mlmodel, expected_type_str="fp16")
        assert_output_dtype(mlmodel, expected_type_str="fp32")
        verify_prediction(mlmodel)

        mlmodel = ct.convert(
            float32_input_model_relu_ops,
            inputs=[ct.TensorType(shape=(10, 20), dtype=np.float16)],
            outputs=[ct.TensorType(dtype=np.float32)],
            minimum_deployment_target=ct.target.macOS13,
        )
        # Two consecutive relus are merged in the `merge_consecutive_relus` pass.
        assert_ops_in_mil_program(mlmodel, expected_op_list=["relu", "cast"])
        assert_input_dtype(mlmodel, expected_type_str="fp16")
        assert_output_dtype(mlmodel, expected_type_str="fp32")
        verify_prediction(mlmodel)

        mlmodel = ct.convert(
            int32_input_model,
            inputs=[ct.TensorType(shape=(10, 20), dtype=np.float16)],
            outputs=[ct.TensorType(dtype=np.float32)],
            minimum_deployment_target=ct.target.macOS13,
        )
        assert_ops_in_mil_program(mlmodel, expected_op_list=["add", "cast"])
        assert_input_dtype(mlmodel, expected_type_str="fp16")
        assert_output_dtype(mlmodel, expected_type_str="fp32")
        verify_prediction(mlmodel)

    def test_fp16_input_dtype_fp32_precision(self, float32_input_model_add_op, float32_input_model_relu_ops,
                                             int32_input_model):
        """
        Same test as test_fp16_input_dtype, but with Float32 precision
        """
        mlmodel = ct.convert(float32_input_model_add_op,
                             inputs=[ct.TensorType(shape=(10, 20), dtype=np.float16)],
                             minimum_deployment_target=ct.target.macOS13,
                             compute_precision=ct.precision.FLOAT32,
                             )
        assert_ops_in_mil_program(mlmodel, expected_op_list=["cast", "add"])
        assert_input_dtype(mlmodel, expected_type_str="fp16")
        assert_output_dtype(mlmodel, expected_type_str="fp32")
        verify_prediction(mlmodel)

        """
        Although no FP16ComputePrecision is applied, the float16 input propagates through the network
        """
        mlmodel = ct.convert(float32_input_model_relu_ops,
                             inputs=[ct.TensorType(shape=(10, 20), dtype=np.float16)],
                             minimum_deployment_target=ct.target.macOS13,
                             compute_precision=ct.precision.FLOAT32,
                             )
        assert_ops_in_mil_program(mlmodel, expected_op_list=["cast", "relu"])
        assert_input_dtype(mlmodel, expected_type_str="fp16")
        assert_output_dtype(mlmodel, expected_type_str="fp32")

    def test_input_name_specified_by_user(self, float32_input_model_relu_ops,
                                          float32_two_input_model):
        mlmodel = ct.convert(float32_input_model_relu_ops,
                             inputs=[ct.TensorType(shape=(10, 20), name="my_custom_input_name")],
                             minimum_deployment_target=ct.target.macOS12)
        assert_input_dtype(mlmodel, expected_type_str="fp32", expected_name="my_custom_input_name")

        mlmodel = ct.convert(float32_two_input_model,
                             inputs=[ct.TensorType(shape=(10, 20), name="user_provided_name_1"),
                                     ct.TensorType(shape=(10, 20), name="user_provided_name_2")],
                             minimum_deployment_target=ct.target.macOS12)
        assert_input_dtype(mlmodel, expected_type_str="fp32", expected_name="user_provided_name_1", index=0)
        assert_input_dtype(mlmodel, expected_type_str="fp32", expected_name="user_provided_name_2", index=1)

    def test_two_input_model(self, float32_two_input_model):
        # test that error is raised if only 1 input is provided
        with pytest.raises(ValueError):
            ct.convert(float32_two_input_model,
                       inputs=[ct.TensorType(shape=(10, 20), dtype=np.int32)],
                       minimum_deployment_target=ct.target.macOS12)


        # test forcing 1st input to type int32
        mlmodel = ct.convert(float32_two_input_model,
                             inputs=[ct.TensorType(shape=(10, 20), dtype=np.int32),
                                     ct.TensorType(shape=(10, 20))],
                             minimum_deployment_target=ct.target.macOS12)
        assert_input_dtype(mlmodel, expected_type_str="int32", index=0)
        assert_input_dtype(mlmodel, expected_type_str="fp32", index=1)
        assert_output_dtype(mlmodel, expected_type_str="fp32")

        # test forcing both inputs to be int32
        mlmodel = ct.convert(float32_two_input_model,
                             inputs=[ct.TensorType(shape=(10, 20), dtype=np.int32),
                                     ct.TensorType(shape=(10, 20), dtype=np.int32),
                                     ],
                             minimum_deployment_target=ct.target.macOS12)
        assert_input_dtype(mlmodel, expected_type_str="int32", index=0)
        assert_input_dtype(mlmodel, expected_type_str="int32", index=1)
        assert_output_dtype(mlmodel, expected_type_str="int32")

        # test forcing both inputs to be float16
        mlmodel = ct.convert(
            float32_two_input_model,
            inputs=[
                ct.TensorType(shape=(10, 20), dtype=np.float16),
                ct.TensorType(shape=(10, 20), dtype=np.float16),
            ],
            outputs=[
                ct.TensorType(dtype=np.float32),
            ],
            minimum_deployment_target=ct.target.macOS13,
        )
        assert_ops_in_mil_program(mlmodel, expected_op_list=["add", "cast"])
        assert_input_dtype(mlmodel, expected_type_str="fp16", index=0)
        assert_input_dtype(mlmodel, expected_type_str="fp16", index=1)
        assert_output_dtype(mlmodel, expected_type_str="fp32")
        verify_prediction(mlmodel)

    def test_output_name_specified_by_user(self, float32_input_model_relu_ops, float32_two_output_model):
        mlmodel = ct.convert(float32_input_model_relu_ops,
                             inputs=[ct.TensorType(shape=(10, 20), name="custom_input_name")],
                             outputs=[ct.TensorType(name="custom_output_name")],
                             minimum_deployment_target=ct.target.macOS12)
        assert_input_dtype(mlmodel, expected_type_str="fp32", expected_name="custom_input_name")
        assert_output_dtype(mlmodel, expected_type_str="fp32", expected_name="custom_output_name")

        mlmodel = ct.convert(float32_two_output_model,
                             inputs=[ct.TensorType(shape=(10, 20), name="custom_input_name")],
                             outputs=[ct.TensorType(name="custom_output1_name"),
                                      ct.TensorType(name="custom_output2_name")],
                             minimum_deployment_target=ct.target.macOS12)
        assert_input_dtype(mlmodel, expected_type_str="fp32", expected_name="custom_input_name")
        assert_output_dtype(mlmodel, expected_type_str="fp32", expected_name="custom_output1_name", index=0)
        assert_output_dtype(mlmodel, expected_type_str="fp32", expected_name="custom_output2_name", index=1)

    def test_single_output_model(self, int32_input_model, float32_input_model_relu_ops):
        # test output type: if not provided, it should be the default which is float32
        mlmodel = ct.convert(
            int32_input_model,
            inputs=[ct.TensorType(shape=(10, 20), dtype=np.float32)],
            outputs=[ct.TensorType(dtype=np.float32)],
            minimum_deployment_target=ct.target.macOS12,
        )
        assert_ops_in_mil_program(mlmodel, expected_op_list=["cast", "add", "cast"])
        assert_input_dtype(mlmodel, expected_type_str="fp32")
        assert_output_dtype(mlmodel, expected_type_str="fp32")

        # test that the output dtype provided by the user is applied during conversion
        mlmodel = ct.convert(
            float32_input_model_relu_ops,
            inputs=[ct.TensorType(shape=(10, 20), dtype=np.float32)],
            outputs=[ct.TensorType(dtype=np.int32)],
            minimum_deployment_target=ct.target.macOS12,
        )
        assert_input_dtype(mlmodel, expected_type_str="fp32")
        assert_output_dtype(mlmodel, expected_type_str="int32")
        assert_ops_in_mil_program(mlmodel, expected_op_list=["cast", "relu", "cast"])

        # test that an error is raised when shape is provided for the output
        with pytest.raises(ValueError):
            mlmodel = ct.convert(int32_input_model,
                                 inputs=[ct.TensorType(shape=(10, 20))],
                                 outputs=[ct.TensorType(dtype=np.float32, shape=(10, 20))],
                                 minimum_deployment_target=ct.target.macOS12)

        # test that output dtype of float16 is rejected when deployment target is low
        with pytest.raises(TypeError,
                           match="float16 dtype for outputs is only supported for deployment target >= iOS16/macOS13"
                           ):
            ct.convert(float32_input_model_relu_ops,
                       inputs=[ct.TensorType(shape=(10, 20))],
                       outputs=[ct.TensorType(dtype=np.float16)],
                       minimum_deployment_target=ct.target.macOS12,
                       )

        # test that output type float16 is applied correctly
        mlmodel = ct.convert(
            float32_input_model_relu_ops,
            inputs=[ct.TensorType(shape=(10, 20), dtype=np.float32)],
            outputs=[ct.TensorType(dtype=np.float16)],
            minimum_deployment_target=ct.target.macOS13,
        )
        assert_output_dtype(mlmodel, expected_type_str="fp16")
        assert_ops_in_mil_program(mlmodel, expected_op_list=["cast", "relu"])

        # test that input and output types float16 are applied correctly
        mlmodel = ct.convert(float32_input_model_relu_ops,
                             inputs=[ct.TensorType(shape=(10, 20), dtype=np.float16)],
                             outputs=[ct.TensorType(dtype=np.float16)],
                             minimum_deployment_target=ct.target.macOS13,
                             )
        assert_input_dtype(mlmodel, expected_type_str="fp16")
        assert_output_dtype(mlmodel, expected_type_str="fp16")
        assert_ops_in_mil_program(mlmodel, expected_op_list=["relu"])
        verify_prediction(mlmodel)

    def test_multi_output_model(self, float32_two_output_model):
        # check that error is raised when only 1 output provided
        with pytest.raises(ValueError, match="Number of outputs provided, 1, "
                                        "do not match the number of outputs detected in the model, 2"):
            ct.convert(float32_two_output_model,
                       inputs=[ct.TensorType(shape=(10, 20))],
                       outputs=[ct.TensorType()],
                       minimum_deployment_target=ct.target.macOS12)

        # set 1 output to float16 and the other to float32
        mlmodel = ct.convert(float32_two_output_model,
                             inputs=[ct.TensorType(shape=(10, 20), dtype=np.float16)],
                             outputs=[ct.TensorType(name="out1", dtype=np.float16),
                                      ct.TensorType(name="out2", dtype=np.float32)],
                             minimum_deployment_target=ct.target.macOS13,
                             )
        assert_cast_ops_count(mlmodel, expected_count=1)
        assert_input_dtype(mlmodel, expected_type_str="fp16")
        assert_output_dtype(mlmodel, expected_type_str="fp16", expected_name="out1" ,index=0)
        assert_output_dtype(mlmodel, expected_type_str="fp32", expected_name="out2", index=1)
        verify_prediction(mlmodel)

    def test_color_input(self, rank4_input_model, rank3_input_model):
        mlmodel = ct.convert(
            rank4_input_model,
            inputs=[ct.ImageType(shape=(1, 3, 10, 20), color_layout=ct.colorlayout.RGB)],
            outputs=[ct.TensorType(dtype=np.float32)],
            minimum_deployment_target=ct.target.macOS13,
        )
        assert_ops_in_mil_program(mlmodel, expected_op_list=["cast", "add", "cast"])
        assert_spec_input_image_type(mlmodel._spec, expected_feature_type=ft.ImageFeatureType.RGB)
        assert_prog_input_type(mlmodel._mil_program, expected_dtype_str="fp32")
        assert_prog_output_type(mlmodel._mil_program, expected_dtype_str="fp32")
        verify_prediction(mlmodel)

        with pytest.raises(ValueError, match="must have rank 4"):
            mlmodel = ct.convert(rank3_input_model,
                                 inputs=[ct.ImageType(shape=(1, 10, 20), color_layout=ct.colorlayout.RGB)],
                                 minimum_deployment_target=ct.target.macOS12,
                                 )

    def test_grayscale_input(self, rank4_input_model, rank3_input_model, rank4_grayscale_input_model):
        with pytest.raises(ValueError, match="must have rank 4"):
            ct.convert(rank3_input_model,
                       inputs=[ct.ImageType(shape=(1, 10, 20), color_layout=ct.colorlayout.GRAYSCALE)],
                       minimum_deployment_target=ct.target.macOS13,
                      )

        # invalid shape
        with pytest.raises(ValueError):
            ct.convert(rank4_input_model,
                       inputs=[ct.ImageType(shape=(1, 3, 10, 20), color_layout=ct.colorlayout.GRAYSCALE)],
                       minimum_deployment_target=ct.target.macOS13,
                       )

        mlmodel = ct.convert(
            rank4_grayscale_input_model,
            inputs=[ct.ImageType(shape=(1, 1, 10, 20), color_layout=ct.colorlayout.GRAYSCALE)],
            outputs=[ct.TensorType(dtype=np.float32)],
            minimum_deployment_target=ct.target.macOS13,
        )
        assert_ops_in_mil_program(mlmodel, expected_op_list=["cast", "add", "cast"])
        assert_spec_input_image_type(mlmodel._spec, expected_feature_type=ft.ImageFeatureType.GRAYSCALE)
        assert_prog_input_type(mlmodel._mil_program, expected_dtype_str="fp32")
        assert_prog_output_type(mlmodel._mil_program, expected_dtype_str="fp32")
        verify_prediction(mlmodel)

        with pytest.raises(TypeError, match="float16 dtype for inputs is only supported for deployment target >= iOS16/macOS13"):
            ct.convert(rank4_grayscale_input_model,
                       inputs=[ct.ImageType(shape=(1, 1, 10, 20),
                                            color_layout=ct.colorlayout.GRAYSCALE_FLOAT16)],
                       minimum_deployment_target=ct.target.macOS12,
                       )

        # test that grayscale_16 raises error when used with neural network
        with pytest.raises(TypeError, match="float16 dtype for inputs is only supported for deployment target >= iOS16/macOS13"):
            ct.convert(rank4_grayscale_input_model,
                       inputs=[ct.ImageType(shape=(1, 1, 10, 20),
                                            color_layout=ct.colorlayout.GRAYSCALE_FLOAT16)],
                      )

        mlmodel = ct.convert(rank4_grayscale_input_model,
                             inputs=[ct.ImageType(shape=(1, 1, 10, 20),
                                                  color_layout=ct.colorlayout.GRAYSCALE_FLOAT16)],
                             outputs=[ct.TensorType(dtype=np.float16)],
                             minimum_deployment_target=ct.target.macOS13,
                             )
        assert_ops_in_mil_program(mlmodel, expected_op_list=["add"])
        assert_spec_input_image_type(mlmodel._spec, expected_feature_type=ft.ImageFeatureType.GRAYSCALE_FLOAT16)
        assert_prog_input_type(mlmodel._mil_program, expected_dtype_str="fp16")
        assert_output_dtype(mlmodel, expected_type_str="fp16")
        verify_prediction(mlmodel)

    def test_color_output(self, rank4_input_model, float32_input_model_add_op):
        # check that an error is raised if the output shape is not of form (1, 3, H, W)
        with pytest.raises(ValueError, match="must have rank 4. Instead it has rank 2"):
            ct.convert(float32_input_model_add_op,
                       inputs=[ct.TensorType(shape=(10, 20))],
                       outputs=[ct.ImageType(color_layout=ct.colorlayout.RGB)],
                       minimum_deployment_target=ct.target.macOS13)

        mlmodel = ct.convert(rank4_input_model,
                             inputs=[ct.ImageType(shape=(1, 3, 10, 20),
                                                  color_layout=ct.colorlayout.BGR)],
                             outputs=[ct.ImageType(color_layout=ct.colorlayout.RGB)],
                             minimum_deployment_target=ct.target.macOS13,
                             )
        assert_ops_in_mil_program(mlmodel, expected_op_list=["cast", "add", "cast"])
        assert_spec_input_image_type(mlmodel._spec, expected_feature_type=ft.ImageFeatureType.BGR)
        assert_spec_output_image_type(mlmodel._spec, expected_feature_type=ft.ImageFeatureType.RGB)
        assert_prog_input_type(mlmodel._mil_program, expected_dtype_str="fp32")
        assert_prog_output_type(mlmodel._mil_program, expected_dtype_str="fp32")
        verify_prediction(mlmodel)

        # check neural network conversion
        mlmodel = ct.convert(
            rank4_input_model,
            inputs=[ct.ImageType(shape=(1, 3, 10, 20), color_layout=ct.colorlayout.RGB)],
            outputs=[ct.ImageType(color_layout=ct.colorlayout.BGR)],
            convert_to="neuralnetwork",
        )
        assert_ops_in_mil_program(mlmodel, expected_op_list=["add"])
        assert_spec_input_image_type(mlmodel._spec, expected_feature_type=ft.ImageFeatureType.RGB)
        assert_spec_output_image_type(mlmodel._spec, expected_feature_type=ft.ImageFeatureType.BGR)
        verify_prediction(mlmodel)

    def test_grayscale_output(self, rank4_grayscale_input_model):
        with pytest.raises(TypeError, match="float16 dtype for outputs is only supported for deployment target >= iOS16/macOS13"):
            ct.convert(rank4_grayscale_input_model,
                       inputs=[ct.TensorType(shape=(1, 1, 10, 20))],
                       outputs=[ct.ImageType(color_layout=ct.colorlayout.GRAYSCALE_FLOAT16)],
                       minimum_deployment_target=ct.target.macOS12,
                      )

        mlmodel = ct.convert(
            rank4_grayscale_input_model,
            inputs=[ct.ImageType(shape=(1, 1, 10, 20), color_layout=ct.colorlayout.GRAYSCALE)],
            outputs=[ct.ImageType(color_layout=ct.colorlayout.GRAYSCALE)],
            convert_to="neuralnetwork",
        )
        assert_ops_in_mil_program(mlmodel, expected_op_list=["add"])
        assert_spec_input_image_type(mlmodel._spec, expected_feature_type=ft.ImageFeatureType.GRAYSCALE)
        assert_spec_output_image_type(mlmodel._spec, expected_feature_type=ft.ImageFeatureType.GRAYSCALE)
        verify_prediction(mlmodel)

        mlmodel = ct.convert(rank4_grayscale_input_model,
                             inputs=[ct.ImageType(shape=(1, 1, 10, 20),
                                                  color_layout=ct.colorlayout.GRAYSCALE_FLOAT16)],
                             outputs=[ct.ImageType(color_layout=ct.colorlayout.GRAYSCALE_FLOAT16)],
                             minimum_deployment_target=ct.target.macOS13,
                             )
        assert_ops_in_mil_program(mlmodel, expected_op_list=["add"])
        assert_spec_input_image_type(mlmodel._spec, expected_feature_type=ft.ImageFeatureType.GRAYSCALE_FLOAT16)
        assert_spec_output_image_type(mlmodel._spec, expected_feature_type=ft.ImageFeatureType.GRAYSCALE_FLOAT16)
        assert_prog_input_type(mlmodel._mil_program, expected_dtype_str="fp16")
        assert_prog_output_type(mlmodel._mil_program, expected_dtype_str="fp16")
        verify_prediction(mlmodel)

        mlmodel = ct.convert(rank4_grayscale_input_model,
                             inputs=[ct.ImageType(shape=(1, 1, 10, 20),
                                                  color_layout=ct.colorlayout.GRAYSCALE)],
                             outputs=[ct.ImageType(color_layout=ct.colorlayout.GRAYSCALE_FLOAT16)],
                             minimum_deployment_target=ct.target.macOS13,
                             )
        assert_ops_in_mil_program(mlmodel, expected_op_list=["cast", "add"])
        assert_spec_input_image_type(mlmodel._spec, expected_feature_type=ft.ImageFeatureType.GRAYSCALE)
        assert_spec_output_image_type(mlmodel._spec, expected_feature_type=ft.ImageFeatureType.GRAYSCALE_FLOAT16)
        assert_prog_input_type(mlmodel._mil_program, expected_dtype_str="fp32")
        assert_prog_output_type(mlmodel._mil_program, expected_dtype_str="fp16")
        verify_prediction(mlmodel)

    def test_linear_model(self, linear_model):
        # this will test the fuse_linear_bias pass, when the inputs are of type float16
        mlmodel = ct.convert(linear_model,
                             inputs=[ct.TensorType(shape=(1, 10), dtype=np.float16)],
                             outputs=[ct.TensorType(dtype=np.float16)],
                             minimum_deployment_target=ct.target.macOS13,
                             )
        assert_input_dtype(mlmodel, expected_type_str="fp16")
        assert_output_dtype(mlmodel, expected_type_str="fp16")
        assert_ops_in_mil_program(mlmodel, ["linear", "relu"])
        verify_prediction(mlmodel)


    def test_classifier(self):
        torch_model = torch.nn.ReLU().eval()
        traced_model = torch.jit.trace(torch_model, torch.rand(3,))
        model = ct.convert(
            traced_model,
            inputs=[ct.TensorType(shape=(3,), dtype=np.float16)],
            outputs=[ct.TensorType(dtype=np.float16)],
            classifier_config = ct.ClassifierConfig(['a', 'b', 'c']),
            convert_to='mlprogram',
            minimum_deployment_target=ct.target.macOS13,
        )
        assert_input_dtype(model, expected_type_str="fp16")
        assert_ops_in_mil_program(model, ["relu", "cast", "classify"])
        spec = model.get_spec()
        input_name = spec.description.input[0].name
        out_dict = model.predict({input_name : np.array([1.0, 2.0, 3.0])})
        assert 'classLabel' in out_dict
        assert out_dict['classLabel'] == 'c'
        assert len(spec.description.output) == 2
        assert "classLabel_probs" in out_dict
        assert isinstance(out_dict["classLabel_probs"], dict)

    def test_prediction_with_fp16_io(self):
        torch_model = torch.nn.Linear(30, 5).eval()
        traced_model = torch.jit.trace(torch_model, torch.rand(1, 30))
        mlmodel = ct.convert(traced_model,
                             inputs=[ct.TensorType(name="input", shape=(1, 30), dtype=np.float32)],
                             outputs=[ct.TensorType(dtype=np.float32)],
                             minimum_deployment_target=ct.target.macOS13,
                             compute_units=ct.ComputeUnit.CPU_ONLY,
                             )
        # test prediction
        sample_input = np.random.rand(1, 30).astype(np.float32) * 10
        model_output = mlmodel.predict({"input": sample_input})[mlmodel._spec.description.output[0].name]
        reference_output = traced_model(torch.from_numpy(sample_input)).detach().numpy()
        np.testing.assert_allclose(reference_output, model_output, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(ct.utils._macos_version() < (13, 0), reason='Tests are for deployment target ios16/macos13')
class TestGrayscaleImagePredictions:

    def test_grayscale_input_image(self, rank4_grayscale_input_model):
        mlmodel = ct.convert(rank4_grayscale_input_model,
                             inputs=[ct.ImageType(name="input_image",
                                                  shape=(1, 1, 10, 20),
                                                  color_layout=ct.colorlayout.GRAYSCALE)],
                             outputs=[ct.TensorType(name="output")],
                             minimum_deployment_target=ct.target.macOS13,
                             )
        sample_input = np.random.randint(low=0, high=246, size=(1, 1, 10, 20))
        img_input = Image.fromarray(sample_input[0, 0, :, :].astype(np.uint8), 'L')
        model_output = mlmodel.predict({"input_image": img_input})['output']
        reference_output = rank4_grayscale_input_model(torch.from_numpy(sample_input.astype(np.float32))).detach().numpy()
        np.testing.assert_allclose(reference_output, model_output, rtol=1e-2, atol=1e-2)

    def test_grayscale_fp16_input_image(self, rank4_grayscale_input_model):
        mlmodel = ct.convert(rank4_grayscale_input_model,
                             inputs=[ct.ImageType(name="input_image",
                                                  shape=(1, 1, 10, 20),
                                                  color_layout=ct.colorlayout.GRAYSCALE_FLOAT16)],
                             outputs=[ct.TensorType(name="output")],
                             minimum_deployment_target=ct.target.macOS13,
                             )

        # incorrect way to do prediction
        with pytest.raises(TypeError,
                           match="must be of type PIL.Image.Image with mode=='F'",
                           ):
            sample_input = np.random.randint(low=0, high=246, size=(1, 1, 10, 20))
            img_input = Image.fromarray(sample_input[0, 0, :, :].astype(np.uint8), 'L')
            mlmodel.predict({"input_image": img_input})

        # correct way to do prediction
        sample_input = np.random.rand(1, 1, 10, 20) # in between [0, 1]
        img_input = Image.fromarray(sample_input[0, 0, :, :].astype(np.float32), 'F')
        model_output = mlmodel.predict({"input_image": img_input})['output']
        reference_output = rank4_grayscale_input_model(torch.from_numpy(sample_input.astype(np.float32))).detach().numpy()
        np.testing.assert_allclose(reference_output, model_output, rtol=1e-2, atol=1e-2)

    def test_grayscale_output_image(self, rank4_grayscale_input_model):
        mlmodel = ct.convert(rank4_grayscale_input_model,
                             inputs=[ct.TensorType(name="input",
                                                  shape=(1, 1, 10, 20))],
                             outputs=[ct.ImageType(name="output_image",
                                                   color_layout=ct.colorlayout.GRAYSCALE)],
                             minimum_deployment_target=ct.target.macOS13,
                             compute_precision=ct.precision.FLOAT32,
                             )
        sample_input = np.random.randint(low=0, high=200, size=(1, 1, 10, 20)).astype(np.float32)
        model_output_pil_image = mlmodel.predict({"input": sample_input})['output_image']
        assert isinstance(model_output_pil_image, Image.Image)
        assert model_output_pil_image.mode == "L"
        model_output_as_numpy = np.array(model_output_pil_image)
        reference_output = rank4_grayscale_input_model(torch.from_numpy(sample_input)).detach().numpy()
        reference_output = np.squeeze(reference_output)
        np.testing.assert_allclose(reference_output, model_output_as_numpy, rtol=1e-2, atol=1e-2)

    def test_grayscale_fp16_output_image(self, rank4_grayscale_input_model):
        mlmodel = ct.convert(rank4_grayscale_input_model,
                             inputs=[ct.TensorType(name="input",
                                                  shape=(1, 1, 10, 20))],
                             outputs=[ct.ImageType(name="output_image",
                                                   color_layout=ct.colorlayout.GRAYSCALE_FLOAT16)],
                             minimum_deployment_target=ct.target.macOS13,
                             compute_precision=ct.precision.FLOAT32,
                             )
        sample_input = np.random.randint(low=0, high=200, size=(1, 1, 10, 20)).astype(np.float32)
        model_output_pil_image = mlmodel.predict({"input": sample_input})['output_image']
        assert isinstance(model_output_pil_image, Image.Image)
        assert model_output_pil_image.mode == "F"
        model_output_as_numpy = np.array(model_output_pil_image)
        reference_output = rank4_grayscale_input_model(torch.from_numpy(sample_input)).detach().numpy()
        reference_output = np.squeeze(reference_output)
        np.testing.assert_allclose(reference_output, model_output_as_numpy, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(
    ct.utils._macos_version() < (14, 0), reason="Tests are for deployment target iOS16/macos14"
)
class TestQuantizationConversionAPI:
    def test_dynamic_quantization(self):
        torch.backends.quantized.engine = "qnnpack"

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = torch.nn.Linear(3, 2)

            def forward(self, x):
                x = self.fc(x)
                return x

        SHAPE = (4, 3)
        x = torch.randn(SHAPE)

        model_fp32 = Model()
        model_int8 = torch.ao.quantization.quantize_dynamic(
            model_fp32,
            {torch.nn.Linear},  # a set of layers to dynamically quantize
            dtype=torch.qint8,
        )
        model_int8.eval()

        traced_model = torch.jit.trace(model_int8, x)

        with pytest.raises(
            RuntimeError,
            match=(
                r"PyTorch convert function for op '.*_dynamic' not implemented\.\n"
                r"Dynamic quantized models are not supported by Core ML.\n"
                r"Please use static quantization or the APIs in coremltools.optimize to quantize/compress models."
            ),
        ):
            ct.convert(traced_model, inputs=[ct.TensorType(shape=SHAPE)])

    def test_static_quantization_as_activation_quantization(self):
        torch.backends.quantized.engine = "qnnpack"

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.quant = torch.ao.quantization.QuantStub()
                self.conv = torch.nn.Conv2d(3, 2, 5)
                self.relu = torch.nn.ReLU()
                self.dequant = torch.ao.quantization.DeQuantStub()

            def forward(self, x):
                x = self.quant(x)
                x = self.conv(x)
                x = self.relu(x)
                x = self.dequant(x)
                return x

        SHAPE = (4, 3, 8, 16)
        x = torch.randn(SHAPE)

        model_fp32 = Model()
        model_fp32.eval()

        model_fp32.qconfig = torch.ao.quantization.get_default_qconfig("qnnpack")
        model_fp32_fused = torch.ao.quantization.fuse_modules(model_fp32, [["conv", "relu"]])
        model_fp32_prepared = torch.ao.quantization.prepare(model_fp32_fused)
        model_fp32_prepared(x)
        model_int8 = torch.ao.quantization.convert(model_fp32_prepared)

        traced_model = torch.jit.trace(model_int8, x)
        coreml_model = ct.convert(
            traced_model,
            inputs=[ct.TensorType(name="x", shape=SHAPE)],
            outputs=[ct.TensorType(name="y")],
            minimum_deployment_target=ct.target.iOS17,
        )

        ops = get_op_types_in_program(coreml_model._mil_program)
        # constexpr_affine_dequantize and cast -> quantize can have arbitrary order
        assert set(ops[:2]) == set(["quantize", "constexpr_affine_dequantize"])

        # these ops have well-defined order
        assert ops[2:] == [
            # quantized ConvRelu op
            "dequantize",
            "conv",
            "relu",
            "quantize",
            # dequantize and output
            "dequantize",
        ]

        output = traced_model(x)
        coreml_output = coreml_model.predict({"x": x})["y"]
        np.testing.assert_allclose(output, coreml_output, rtol=1e-2, atol=2e-2)

    def test_static_quantization_as_weight_compression(self):
        torch.backends.quantized.engine = "qnnpack"

        weight = torch.rand(5, 3, 2, 4)

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.quant = torch.ao.quantization.QuantStub()
                self.dequant = torch.ao.quantization.DeQuantStub()

            def forward(self, x):
                quantized_weight = self.quant(weight)
                dequantized_weight = self.dequant(quantized_weight)
                y = torch.nn.functional.conv2d(x, dequantized_weight)
                return y

        SHAPE = (4, 3, 16, 32)
        x = torch.randn(SHAPE)

        model_fp32 = Model()
        model_fp32.eval()

        model_fp32.qconfig = torch.ao.quantization.get_default_qconfig("qnnpack")
        model_fp32_prepared = torch.ao.quantization.prepare(model_fp32)
        model_fp32_prepared(x)
        model_int8 = torch.ao.quantization.convert(model_fp32_prepared)

        traced_model = torch.jit.trace(model_int8, x)
        coreml_model = ct.convert(
            traced_model,
            inputs=[ct.TensorType(name="x", shape=SHAPE)],
            outputs=[ct.TensorType(name="y")],
            minimum_deployment_target=ct.target.iOS17,
        )

        ops = get_op_types_in_program(coreml_model._mil_program)
        # constexpr_affine_dequantize and cast can have arbitrary order
        assert ops == [
            "constexpr_affine_dequantize",
            "conv",
        ]
        output = traced_model(x)
        coreml_output = coreml_model.predict({"x": x})["y"]
        np.testing.assert_allclose(output, coreml_output, rtol=1e-2, atol=2e-2)


class TestiOS16DefaultIODtype:
    """
    This class tests the default i/o dtype behavior for iOS16 (and above) models.
    """

    @staticmethod
    def _verify_model_io(mlmodel, input_dtype, output_dtype, expected_op_list):
        """
        This utility function verifies the model's i/o dtypes and expected ops
        """
        assert_input_dtype(mlmodel, expected_type_str=input_dtype)
        assert_output_dtype(mlmodel, expected_type_str=output_dtype)
        assert_ops_in_mil_program(mlmodel, expected_op_list=expected_op_list)
        verify_prediction(mlmodel)

    def test_iO16_default_fp16_input(self, float32_input_model_add_op):
        """
        With minimum_deployment_target set >= iOS16, and if the compute precision is
        set to fp16. By default, a fp16 i/o model is produced.
        However, if the users specify the dtype, the converter is going to respect that.
        """
        # Case 1: Inputs given / outputs None
        mlmodel = ct.convert(
            float32_input_model_add_op,
            inputs=[ct.TensorType(shape=(10, 20))],
            minimum_deployment_target=ct.target.iOS16,
        )
        self._verify_model_io(
            mlmodel,
            input_dtype="fp16",
            output_dtype="fp16",
            expected_op_list=["add"],
        )

        # Case 2: Inputs given / outputs given
        mlmodel = ct.convert(
            float32_input_model_add_op,
            inputs=[ct.TensorType(shape=(10, 20), dtype=None)],
            outputs=[ct.TensorType(dtype=None)],
            minimum_deployment_target=ct.target.iOS16,
        )
        self._verify_model_io(
            mlmodel,
            input_dtype="fp16",
            output_dtype="fp16",
            expected_op_list=["add"],
        )

        # Case 3: Inputs set fp32 / outputs None
        mlmodel = ct.convert(
            float32_input_model_add_op,
            inputs=[ct.TensorType(shape=(10, 20), dtype=np.float32)],
            minimum_deployment_target=ct.target.iOS16,
        )
        self._verify_model_io(
            mlmodel,
            input_dtype="fp32",
            output_dtype="fp16",
            expected_op_list=["cast", "add"],
        )

        # Case 4: Inputs set fp32 / outputs given
        mlmodel = ct.convert(
            float32_input_model_add_op,
            inputs=[ct.TensorType(shape=(10, 20), dtype=np.float32)],
            outputs=[ct.TensorType(dtype=None)],
            minimum_deployment_target=ct.target.iOS16,
        )
        self._verify_model_io(
            mlmodel,
            input_dtype="fp32",
            output_dtype="fp16",
            expected_op_list=["cast", "add"],
        )

        # Case 5: Inputs given / outputs set to fp32
        mlmodel = ct.convert(
            float32_input_model_add_op,
            inputs=[ct.TensorType(shape=(10, 20))],
            outputs=[ct.TensorType(dtype=np.float32)],
            minimum_deployment_target=ct.target.iOS16,
        )
        self._verify_model_io(
            mlmodel,
            input_dtype="fp16",
            output_dtype="fp32",
            expected_op_list=["add", "cast"],
        )

        # Case 6: Inputs / outputs both set to fp32
        mlmodel = ct.convert(
            float32_input_model_add_op,
            inputs=[ct.TensorType(shape=(10, 20), dtype=np.float32)],
            outputs=[ct.TensorType(dtype=np.float32)],
            minimum_deployment_target=ct.target.iOS16,
        )
        self._verify_model_io(
            mlmodel,
            input_dtype="fp32",
            output_dtype="fp32",
            expected_op_list=["cast", "add", "cast"],
        )

    def test_iO16_default_fp16_io_with_multiple_inputs(self, float32_two_input_model):
        """
        For the multiple inputs model, the converter only set the default dtype for
        inputs with unspecified dtype.
        """
        # Case 1: first input is set to fp32
        mlmodel = ct.convert(
            float32_two_input_model,
            inputs=[ct.TensorType(shape=(10, 20), dtype=np.float32), ct.TensorType(shape=(10, 20))],
            minimum_deployment_target=ct.target.iOS16,
        )
        assert_input_dtype(mlmodel, expected_type_str="fp32", index=0)
        assert_input_dtype(mlmodel, expected_type_str="fp16", index=1)
        assert_output_dtype(mlmodel, expected_type_str="fp16")
        assert_ops_in_mil_program(mlmodel, expected_op_list=["cast", "add"])

        # Case 2: second input is set to fp32
        mlmodel = ct.convert(
            float32_two_input_model,
            inputs=[ct.TensorType(shape=(10, 20)), ct.TensorType(shape=(10, 20), dtype=np.float32)],
            minimum_deployment_target=ct.target.iOS16,
        )
        assert_input_dtype(mlmodel, expected_type_str="fp16", index=0)
        assert_input_dtype(mlmodel, expected_type_str="fp32", index=1)
        assert_output_dtype(mlmodel, expected_type_str="fp16")
        assert_ops_in_mil_program(mlmodel, expected_op_list=["cast", "add"])

        # Case 3: both inputs are set to fp32
        mlmodel = ct.convert(
            float32_two_input_model,
            inputs=[
                ct.TensorType(shape=(10, 20), dtype=np.float32),
                ct.TensorType(shape=(10, 20), dtype=np.float32),
            ],
            minimum_deployment_target=ct.target.iOS16,
        )
        assert_input_dtype(mlmodel, expected_type_str="fp32", index=0)
        assert_input_dtype(mlmodel, expected_type_str="fp32", index=1)
        assert_output_dtype(mlmodel, expected_type_str="fp16")
        assert_ops_in_mil_program(mlmodel, expected_op_list=["cast", "cast", "add"])

        # Case 4: both inputs are not set
        mlmodel = ct.convert(
            float32_two_input_model,
            inputs=[ct.TensorType(shape=(10, 20)), ct.TensorType(shape=(10, 20))],
            minimum_deployment_target=ct.target.iOS16,
        )
        assert_input_dtype(mlmodel, expected_type_str="fp16", index=0)
        assert_input_dtype(mlmodel, expected_type_str="fp16", index=1)
        assert_output_dtype(mlmodel, expected_type_str="fp16")
        assert_ops_in_mil_program(mlmodel, expected_op_list=["add"])

    def test_iO16_default_fp16_io_with_multiple_outputs(
        self, float32_two_output_model, int32_float32_two_output_model
    ):
        """
        For the multiple outputs model, the converter only set the default dtype to  fp16 for
        outputs that satisfy
        1. dtype is None
        2. inferred dtype is fp32
        """
        # Case 1: first output is set to fp32
        mlmodel = ct.convert(
            float32_two_output_model,
            inputs=[ct.TensorType(shape=(10, 20))],
            outputs=[ct.TensorType(dtype=np.float32), ct.TensorType(dtype=None)],
            minimum_deployment_target=ct.target.iOS16,
        )
        assert_input_dtype(mlmodel, expected_type_str="fp16")
        assert_output_dtype(mlmodel, expected_type_str="fp32", index=0)
        assert_output_dtype(mlmodel, expected_type_str="fp16", index=1)
        assert_ops_in_mil_program(mlmodel, expected_op_list=["relu", "clip", "cast"])

        # Case 2: second output is set to fp32
        mlmodel = ct.convert(
            float32_two_output_model,
            inputs=[ct.TensorType(shape=(10, 20))],
            outputs=[ct.TensorType(dtype=None), ct.TensorType(dtype=np.float32)],
            minimum_deployment_target=ct.target.iOS16,
        )
        assert_input_dtype(mlmodel, expected_type_str="fp16")
        assert_output_dtype(mlmodel, expected_type_str="fp16", index=0)
        assert_output_dtype(mlmodel, expected_type_str="fp32", index=1)
        assert_ops_in_mil_program(mlmodel, expected_op_list=["relu", "clip", "cast"])

        # Case 3: both outputs are set to fp32
        mlmodel = ct.convert(
            float32_two_output_model,
            inputs=[ct.TensorType(shape=(10, 20))],
            outputs=[ct.TensorType(dtype=np.float32), ct.TensorType(dtype=np.float32)],
            minimum_deployment_target=ct.target.iOS16,
        )
        assert_input_dtype(mlmodel, expected_type_str="fp16")
        assert_output_dtype(mlmodel, expected_type_str="fp32", index=0)
        assert_output_dtype(mlmodel, expected_type_str="fp32", index=1)
        assert_ops_in_mil_program(mlmodel, expected_op_list=["relu", "clip", "cast", "cast"])

        # Case 4: both outputs are not set
        mlmodel = ct.convert(
            float32_two_output_model,
            inputs=[ct.TensorType(shape=(10, 20))],
            outputs=[ct.TensorType(dtype=None), ct.TensorType(dtype=None)],
            minimum_deployment_target=ct.target.iOS16,
        )
        assert_input_dtype(mlmodel, expected_type_str="fp16")
        assert_output_dtype(mlmodel, expected_type_str="fp16", index=0)
        assert_output_dtype(mlmodel, expected_type_str="fp16", index=1)
        assert_ops_in_mil_program(mlmodel, expected_op_list=["relu", "clip"])

        # Case 5: outputs is not provided at all
        mlmodel = ct.convert(
            float32_two_output_model,
            inputs=[ct.TensorType(shape=(10, 20))],
            minimum_deployment_target=ct.target.iOS16,
        )
        assert_input_dtype(mlmodel, expected_type_str="fp16")
        assert_output_dtype(mlmodel, expected_type_str="fp16", index=0)
        assert_output_dtype(mlmodel, expected_type_str="fp16", index=1)
        assert_ops_in_mil_program(mlmodel, expected_op_list=["relu", "clip"])

        # Case 6: int32 and fp32 output. The fp32 defaults to fp32 while the int32 one remains unchanged.
        mlmodel = ct.convert(
            int32_float32_two_output_model,
            inputs=[
                ct.TensorType(shape=(10, 20), dtype=np.int32),
                ct.TensorType(shape=(10, 20), dtype=np.float32),
            ],
            minimum_deployment_target=ct.target.iOS16,
        )
        assert_input_dtype(mlmodel, expected_type_str="int32", index=0)
        assert_input_dtype(mlmodel, expected_type_str="fp32", index=1)
        assert_output_dtype(mlmodel, expected_type_str="int32", index=0)
        assert_output_dtype(mlmodel, expected_type_str="fp16", index=1)
        assert_ops_in_mil_program(mlmodel, expected_op_list=["add", "cast", "add"])

        # Case 7: int32 and fp32 output. The fp32 defaults to fp32 while the int32 one remains unchanged.
        mlmodel = ct.convert(
            int32_float32_two_output_model,
            inputs=[
                ct.TensorType(shape=(10, 20), dtype=np.int32),
                ct.TensorType(shape=(10, 20)),
            ],
            minimum_deployment_target=ct.target.iOS16,
        )
        assert_input_dtype(mlmodel, expected_type_str="int32", index=0)
        assert_input_dtype(mlmodel, expected_type_str="fp16", index=1)
        assert_output_dtype(mlmodel, expected_type_str="int32", index=0)
        assert_output_dtype(mlmodel, expected_type_str="fp16", index=1)
        assert_ops_in_mil_program(mlmodel, expected_op_list=["add", "add"])

        # Case 8: int32 and fp32 output. The fp32 defaults to fp32 while the int32 one remains unchanged.
        mlmodel = ct.convert(
            int32_float32_two_output_model,
            inputs=[
                ct.TensorType(shape=(10, 20), dtype=np.int32),
                ct.TensorType(shape=(10, 20)),
            ],
            outputs=[
                ct.TensorType(name="out1"),
                ct.TensorType(name="out2"),
            ],
            minimum_deployment_target=ct.target.iOS16,
        )
        assert_input_dtype(mlmodel, expected_type_str="int32", index=0)
        assert_input_dtype(mlmodel, expected_type_str="fp16", index=1)
        assert_output_dtype(mlmodel, expected_type_str="int32", index=0)
        assert_output_dtype(mlmodel, expected_type_str="fp16", index=1)
        assert_ops_in_mil_program(mlmodel, expected_op_list=["add", "add"])

        # Case 9: two int32 outputs. Nothing changed.
        mlmodel = ct.convert(
            int32_float32_two_output_model,
            inputs=[
                ct.TensorType(shape=(10, 20), dtype=np.int32),
                ct.TensorType(shape=(10, 20), dtype=np.int32),
            ],
            minimum_deployment_target=ct.target.iOS16,
        )
        assert_input_dtype(mlmodel, expected_type_str="int32", index=0)
        assert_input_dtype(mlmodel, expected_type_str="int32", index=1)
        assert_output_dtype(mlmodel, expected_type_str="int32", index=0)
        assert_output_dtype(mlmodel, expected_type_str="int32", index=1)
        assert_ops_in_mil_program(mlmodel, expected_op_list=["add", "add"])

    def test_iO16_default_image_dtype_input(
        self,
        rank4_input_model,
        rank4_grayscale_input_model,
    ):
        """
        We keep the input dtype for the image input model to fp32, unless it is GRAYSCALE_FLOAT16
        """
        # Example 1
        mlmodel = ct.convert(
            rank4_input_model,
            inputs=[ct.ImageType(shape=(1, 3, 10, 20), color_layout=ct.colorlayout.RGB)],
            minimum_deployment_target=ct.target.iOS16,
        )
        assert_spec_input_image_type(mlmodel._spec, expected_feature_type=ft.ImageFeatureType.RGB)
        assert_prog_input_type(mlmodel._mil_program, expected_dtype_str="fp32")
        assert_prog_output_type(mlmodel._mil_program, expected_dtype_str="fp16")
        verify_prediction(mlmodel)

        # Example 2
        mlmodel = ct.convert(
            rank4_input_model,
            inputs=[ct.ImageType(shape=(1, 3, 10, 20), color_layout=ct.colorlayout.BGR)],
            minimum_deployment_target=ct.target.iOS16,
        )
        assert_spec_input_image_type(mlmodel._spec, expected_feature_type=ft.ImageFeatureType.BGR)
        assert_prog_input_type(mlmodel._mil_program, expected_dtype_str="fp32")
        assert_prog_output_type(mlmodel._mil_program, expected_dtype_str="fp16")
        verify_prediction(mlmodel)

        # Example 3
        mlmodel = ct.convert(
            rank4_grayscale_input_model,
            inputs=[ct.ImageType(shape=(1, 1, 10, 20), color_layout=ct.colorlayout.GRAYSCALE)],
            minimum_deployment_target=ct.target.iOS16,
        )
        assert_spec_input_image_type(
            mlmodel._spec, expected_feature_type=ft.ImageFeatureType.GRAYSCALE
        )
        assert_prog_input_type(mlmodel._mil_program, expected_dtype_str="fp32")
        assert_prog_output_type(mlmodel._mil_program, expected_dtype_str="fp16")
        verify_prediction(mlmodel)

        # Example 4
        mlmodel = ct.convert(
            rank4_grayscale_input_model,
            inputs=[
                ct.ImageType(shape=(1, 1, 10, 20), color_layout=ct.colorlayout.GRAYSCALE_FLOAT16)
            ],
            minimum_deployment_target=ct.target.iOS16,
        )
        assert_spec_input_image_type(
            mlmodel._spec, expected_feature_type=ft.ImageFeatureType.GRAYSCALE_FLOAT16
        )
        assert_prog_input_type(mlmodel._mil_program, expected_dtype_str="fp16")
        assert_prog_output_type(mlmodel._mil_program, expected_dtype_str="fp16")
        verify_prediction(mlmodel)

    def test_iO16_default_image_dtype_output(
        self,
        rank4_input_model,
        rank4_grayscale_input_model,
    ):
        """
        We keep the output dtype for the image input model to fp32, unless it is GRAYSCALE_FLOAT16
        """
        # Example 1
        mlmodel = ct.convert(
            rank4_input_model,
            inputs=[ct.TensorType(shape=(1, 3, 10, 20))],
            outputs=[ct.ImageType(color_layout=ct.colorlayout.RGB)],
            minimum_deployment_target=ct.target.iOS16,
        )
        assert_prog_input_type(mlmodel._mil_program, expected_dtype_str="fp16")
        assert_prog_output_type(mlmodel._mil_program, expected_dtype_str="fp32")
        assert_spec_output_image_type(mlmodel._spec, expected_feature_type=ft.ImageFeatureType.RGB)
        verify_prediction(mlmodel)

        # Example 2
        mlmodel = ct.convert(
            rank4_input_model,
            inputs=[ct.TensorType(shape=(1, 3, 10, 20))],
            outputs=[ct.ImageType(color_layout=ct.colorlayout.BGR)],
            minimum_deployment_target=ct.target.iOS16,
        )
        assert_prog_input_type(mlmodel._mil_program, expected_dtype_str="fp16")
        assert_prog_output_type(mlmodel._mil_program, expected_dtype_str="fp32")
        assert_spec_output_image_type(mlmodel._spec, expected_feature_type=ft.ImageFeatureType.BGR)
        verify_prediction(mlmodel)

        # Example 3
        mlmodel = ct.convert(
            rank4_grayscale_input_model,
            inputs=[ct.TensorType(shape=(1, 1, 10, 20))],
            outputs=[ct.ImageType(color_layout=ct.colorlayout.GRAYSCALE)],
            minimum_deployment_target=ct.target.iOS16,
        )
        assert_prog_input_type(mlmodel._mil_program, expected_dtype_str="fp16")
        assert_prog_output_type(mlmodel._mil_program, expected_dtype_str="fp32")
        assert_spec_output_image_type(
            mlmodel._spec, expected_feature_type=ft.ImageFeatureType.GRAYSCALE
        )
        verify_prediction(mlmodel)

        # Example 4
        mlmodel = ct.convert(
            rank4_grayscale_input_model,
            inputs=[ct.TensorType(shape=(1, 1, 10, 20))],
            outputs=[ct.ImageType(color_layout=ct.colorlayout.GRAYSCALE_FLOAT16)],
            minimum_deployment_target=ct.target.iOS16,
        )
        assert_prog_input_type(mlmodel._mil_program, expected_dtype_str="fp16")
        assert_prog_output_type(mlmodel._mil_program, expected_dtype_str="fp16")
        assert_spec_output_image_type(
            mlmodel._spec, expected_feature_type=ft.ImageFeatureType.GRAYSCALE_FLOAT16
        )
        verify_prediction(mlmodel)

    def test_iO16_default_fp32_io(self, float32_input_model_add_op):
        """
        With minimum_deployment_target set >= iOS16, and if the compute precision is
        set to fp32. By default, a fp32 i/o model is produced.
        """
        # Case 1: Inputs given / outputs None
        mlmodel = ct.convert(
            float32_input_model_add_op,
            inputs=[ct.TensorType(shape=(10, 20))],
            compute_precision=ct.precision.FLOAT32,
            minimum_deployment_target=ct.target.iOS16,
        )
        self._verify_model_io(
            mlmodel,
            input_dtype="fp32",
            output_dtype="fp32",
            expected_op_list=["add"],
        )

        # Case 2: Inputs given / outputs given
        mlmodel = ct.convert(
            float32_input_model_add_op,
            inputs=[ct.TensorType(shape=(10, 20), dtype=None)],
            outputs=[ct.TensorType(dtype=None)],
            compute_precision=ct.precision.FLOAT32,
            minimum_deployment_target=ct.target.iOS16,
        )
        self._verify_model_io(
            mlmodel,
            input_dtype="fp32",
            output_dtype="fp32",
            expected_op_list=["add"],
        )
