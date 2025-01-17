#  Copyright (c) 2022, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import itertools
import os
import platform
import shutil
import tempfile
from unittest.mock import patch

import numpy as np
import pytest
from packaging.version import Version
from PIL import Image

import coremltools as ct
from coremltools._deps import (
    _HAS_HF,
    _HAS_TORCH,
    _HAS_TORCHAO,
    MSG_TORCH_NOT_FOUND,
    MSG_TORCHAO_NOT_FOUND,
)
from coremltools.converters.mil.frontend.torch.test.testing_utils import _copy_input_data
from coremltools.converters.mil.frontend.torch.torch_op_registry import (
    _TORCH_OPS_REGISTRY,
    TorchOpsRegistry,
    register_torch_op,
)
from coremltools.converters.mil.mil.types.symbolic import any_symbolic
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
from coremltools.models import _METADATA_SOURCE_DIALECT
from coremltools.proto import FeatureTypes_pb2 as ft
from coremltools.test.api.test_api_examples import TestInputs as _TestInputs

if _HAS_TORCH:
    import torch
    import torch.nn as nn
    import torchvision

    torch.manual_seed(1818)

if _HAS_HF:
    from peft import LoraConfig, get_peft_model

if _HAS_TORCHAO:
    from torchao.quantization import quant_api
    from torchao.utils import unwrap_tensor_subclass

@pytest.fixture
def torch_model():
    class TestModule(torch.nn.Module):
        def __init__(self):
            super(TestModule, self).__init__()
            self.linear = torch.nn.Linear(10, 20)

        def forward(self, x):
            return self.linear(x)

    model = TestModule()
    model.eval()
    return model


@pytest.mark.skipif(not _HAS_TORCH, reason=MSG_TORCH_NOT_FOUND)
class TestTorchScriptValidation:
    @staticmethod
    @pytest.mark.parametrize(
        "backend",
        backends,
    )
    def test_no_inputs(torch_model, backend):

        traced_torch_model = torch.jit.trace(torch_model, torch.rand(1, 10))
        with pytest.raises(
            ValueError, match=r'Expected argument "inputs" for TorchScript models not provided'
        ):
            ct.convert(traced_torch_model, convert_to=backend[0])

    @staticmethod
    @pytest.mark.parametrize(
        "backend",
        backends,
    )
    def test_pth_extension(torch_model, tmpdir, backend):
        # test for issue: https://github.com/apple/coremltools/issues/917

        shape = (1, 10)
        traced_torch_model = torch.jit.trace(torch_model, torch.rand(*shape))

        model_path = os.path.join(str(tmpdir), "torch_model.pth")
        traced_torch_model.save(model_path)

        ct.convert(
            model_path,
            source="pytorch",
            inputs=[
                ct.TensorType(
                    shape=shape,
                )
            ],
            convert_to=backend[0],
        )

    @staticmethod
    @pytest.mark.parametrize(
        "backend",
        backends,
    )
    def test_source_dialect_metadata(torch_model, backend):
        shape = (1, 10)
        traced_torch_model = torch.jit.trace(torch_model, torch.rand(*shape))

        mlmodel = ct.convert(
            traced_torch_model,
            source="pytorch",
            inputs=[
                ct.TensorType(
                    shape=shape,
                )
            ],
            convert_to=backend[0],
        )

        assert _METADATA_SOURCE_DIALECT in mlmodel.user_defined_metadata

        assert mlmodel.user_defined_metadata[_METADATA_SOURCE_DIALECT] == "TorchScript"


@pytest.mark.skipif(not _HAS_TORCH, reason=MSG_TORCH_NOT_FOUND)
class TestTorchOpsRegistry:
    @staticmethod
    def test_api_example():
        # Example code in https://apple.github.io/coremltools/docs-guides/source/composite-operators.html#using-composite-ops-with-pytorch-conversion
        # Whenever this test fails, we should update API documentations
        # This test needs to be modified after rdar://117502178 ([Infra][Pytorch] We should deprecate the direct use of _TORCH_OPS_REGISTRY in 7.2)
        from coremltools.converters.mil import Builder as mb
        from coremltools.converters.mil.frontend.torch.ops import _get_inputs
        from coremltools.converters.mil.frontend.torch.torch_op_registry import (
            _TORCH_OPS_REGISTRY,
            register_torch_op,
        )

        default_func = _TORCH_OPS_REGISTRY.get_func("selu")

        # Test ``__contains__`` and ``__delitem__``
        assert "selu" in _TORCH_OPS_REGISTRY
        if "selu" in _TORCH_OPS_REGISTRY:
            del _TORCH_OPS_REGISTRY["selu"]
        assert not "selu" in _TORCH_OPS_REGISTRY

        # Test ``@register_torch_op`` decorator
        @register_torch_op
        def selu(context, node):
            x = _get_inputs(context, node, expected=1)[0]
            x = mb.elu(x=x, alpha=1.6732632423543772)
            x = mb.mul(x=x, y=1.0507009873554805, name=node.name)
            context.add(x)

        # Test ``__getitem__``
        assert _TORCH_OPS_REGISTRY["selu"] is not None

        # Test ``__setitem__``
        _TORCH_OPS_REGISTRY["selu"] = default_func

    @staticmethod
    def test_register_torch_op():
        # Test ``register_torch_op`` works
        def test_func_dummy(context, inputs):
            return
        register_torch_op(test_func_dummy)
        assert _TORCH_OPS_REGISTRY.name_to_func_mapping["test_func_dummy"] is test_func_dummy

        # Test error out for duplicate registration
        with pytest.raises(ValueError, match="Torch op test_func_dummy already registered."):
            register_torch_op(test_func_dummy)

        # Test we can override the function
        def test_func_dummy(context, inputs):
            dummy = 1
            return
        register_torch_op(test_func_dummy, override=True)
        assert _TORCH_OPS_REGISTRY.name_to_func_mapping["test_func_dummy"] is test_func_dummy

        # Cleanup the test
        del _TORCH_OPS_REGISTRY.name_to_func_mapping["test_func_dummy"]


@pytest.mark.skipif(not _HAS_TORCH, reason=MSG_TORCH_NOT_FOUND)
class TestFxNodeSupport:
    """
    The API ``ct.converters.mil.frontend.torch.is_torch_fx_node_supported`` is used
    by 3rd-party code ExecuTorch: https://github.com/pytorch/executorch/pull/1415,
    so we cannot break it
    """

    @staticmethod
    def test_simple_case():
        class Model(torch.nn.Module):
            def forward(self, a, x, b):
                y = torch.mm(a, x)
                z = y + b
                a.sub_(z)
                y = torch.mm(a, x)
                z = y + b
                return z

        model = Model()
        model.eval()
        symbolic_traced = torch.fx.symbolic_trace(model)

        for node in symbolic_traced.graph.nodes:
            # There are many types of torch fx node,
            # we only support "call_function" node for now
            if node.op == "call_function":
                # All PyTorch ops in the example model are supported, so they should all return true
                assert ct.converters.mil.frontend.torch.is_torch_fx_node_supported(node)
            # Other types of torch fx node are not supported
            else:
                assert not ct.converters.mil.frontend.torch.is_torch_fx_node_supported(node)

    @staticmethod
    def test_unsupported_op():
        class Model(torch.nn.Module):
            def forward(self, x, y):
                z = x + y
                return torch.nn.functional.softmax(z)

        model = Model()
        model.eval()
        symbolic_traced = torch.fx.symbolic_trace(model)

        # Mock our torch ops registry, pretending that only "add" is supported
        with patch.object(
            TorchOpsRegistry,
            "__contains__",
            side_effect=(lambda op_name: op_name == "add"),
        ):
            for node in symbolic_traced.graph.nodes:
                # There are many types of torch fx node,
                # we only support "call_function" node for now
                if node.op == "call_function":
                    # Only "add" is supported
                    assert (
                        (node.target.__name__.lower() == "add")
                        == ct.converters.mil.frontend.torch.is_torch_fx_node_supported(node)
                    )
                # Other types of torch fx node are not supported
                else:
                    assert not ct.converters.mil.frontend.torch.is_torch_fx_node_supported(node)


#################################################################################
# Note: Starting from here, all of the following tests are also used as examples
# in https://coremltools.readme.io/docs as a reference.
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
    def _get_classifier_model():
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

        return traced_model, example_input

    @staticmethod
    def _convert_classifier_model(traced_model, example_input, class_type, backend="mlprogram"):
        label = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        if class_type == "str":
            label = list(map(lambda x: str(x), label))
        classifier_config = ct.ClassifierConfig(label)
        return ct.convert(
            traced_model,
            source="pytorch",
            convert_to=backend,
            inputs=[
                ct.TensorType(
                    name="input",
                    shape=example_input.shape,
                    dtype=example_input.numpy().dtype,
                )
            ],
            classifier_config=classifier_config,
        )

    @staticmethod
    def test_torch_classifier():
        def _test_classifier(traced_model, example_input, class_type, backend):
            mlmodel = TestPyTorchConverterExamples._convert_classifier_model(
                traced_model,
                example_input,
                class_type,
                backend,
            )
            if ct.utils._is_macos():
                coreml_out = mlmodel.predict({"input": example_input.detach().numpy()})
                assert "classLabel" in coreml_out
                key_type = str if class_type == "str" else int
                assert isinstance(coreml_out["classLabel"], key_type)

        for class_type in ("str", "int"):
            traced_model, example_input = TestPyTorchConverterExamples._get_classifier_model()
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

    @staticmethod
    @pytest.mark.skipif(
        ct.utils._macos_version() < (15, 0), reason="Tests are for deployment target iOS18/macos15"
    )
    @pytest.mark.xfail(
        reason="rdar://131396853 Lora Adapted Model Dies as ct.models.MLModel but Passes coremltest",
        run=False,
    )
    def test_multifunction_example():
        # util to add adapters
        def adapt_model_with_lora(model):
            lora_config = LoraConfig(
                target_modules=["linear1", "linear2"], r=32, lora_alpha=1
            )  # rank 32
            adapted_model = get_peft_model(model, lora_config)
            return adapted_model

        # define the base model
        class Base(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(6000, 6000)
                self.relu = nn.ReLU()
                self.linear2 = nn.Linear(6000, 6000)

            def forward(self, x):
                x = self.linear1(x)
                x = self.relu(x)
                x = self.linear2(x)
                return x

        base_model = Base()

        # create tmp paths for models
        mlmodel_1_path = tempfile.mkdtemp(suffix=".mlpackage")
        mlmodel_2_path = tempfile.mkdtemp(suffix=".mlpackage")
        multifunction_model_path = tempfile.mkdtemp(suffix=".mlpackage")

        try:
            # first model with adapter
            adapted_model_1 = adapt_model_with_lora(base_model)
            mlmodel_1 = ct.convert(
                torch.jit.trace(adapted_model_1.eval(), torch.rand(1, 6000)),
                inputs=[ct.TensorType(name="input_adpated_model_1", shape=(1, 6000))],
                outputs=[ct.TensorType(name="out_adpated_model_1")],
                minimum_deployment_target=ct.target.iOS18,
                skip_model_load=True,
            )
            mlmodel_1.save(mlmodel_1_path)

            # second model
            adapted_model_2 = adapt_model_with_lora(base_model)
            mlmodel_2 = ct.convert(
                torch.jit.trace(adapted_model_2.eval(), torch.rand(1, 6000)),
                inputs=[ct.TensorType(name="input_adpated_model_2", shape=(1, 6000))],
                outputs=[ct.TensorType(name="out_adpated_model_2")],
                minimum_deployment_target=ct.target.iOS18,
                skip_model_load=True,
            )
            mlmodel_2.save(mlmodel_2_path)

            # combine two models into a multifunction model
            desc = ct.utils.MultiFunctionDescriptor()
            desc.add_function(
                mlmodel_1_path, src_function_name="main", target_function_name="adapter_1"
            )
            desc.add_function(
                mlmodel_2_path, src_function_name="main", target_function_name="adapter_2"
            )
            desc.default_function_name = "adapter_1"
            ct.utils.save_multifunction(desc, multifunction_model_path)

            if platform.machine() == "arm64":
                # The following model fails to run on Intel machines,
                # tracked by rdar://132919101 ([Bug] Intel machines fails on running several multifunction unittest)

                # run the prediction
                mlmodel_1 = ct.models.MLModel(multifunction_model_path)  # Uses default function
                y_1 = mlmodel_1.predict({"input_adpated_model_1": np.random.rand(1, 6000)})

                mlmodel_2 = ct.models.MLModel(multifunction_model_path, function_name="adapter_2")
                y_2 = mlmodel_2.predict({"input_adpated_model_2": np.random.rand(1, 6000)})

                # run the model using CompiledMLModel
                compile_model = ct.models.CompiledMLModel(multifunction_model_path)
                y_1 = mlmodel_1.predict({"input_adpated_model_1": np.random.rand(1, 6000)})

        except:
            raise ValueError("Test failing for test_multifunction_example.")

        finally:
            # cleanup
            shutil.rmtree(mlmodel_1_path)
            shutil.rmtree(mlmodel_2_path)
            shutil.rmtree(multifunction_model_path)

    @staticmethod
    @pytest.mark.skipif(
        ct.utils._macos_version() < (15, 0), reason="Tests are for deployment target iOS18/macos15"
    )
    def test_stateful_accumulator():
        # stateful model definition in torch
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("accumulator", torch.tensor(np.array([0], dtype=np.float16)))

            def forward(self, x):
                self.accumulator += x
                return self.accumulator * self.accumulator

        # convert the trace model into stateful mlmodel
        traced_model = torch.jit.trace(Model().eval(), torch.tensor([1]))
        mlmodel = ct.convert(
            traced_model,
            inputs=[ct.TensorType(shape=(1,))],
            outputs=[ct.TensorType(name="y")],
            states=[
                ct.StateType(
                    wrapped_type=ct.TensorType(
                        shape=(1,),
                    ),
                    name="accumulator",
                ),
            ],
            minimum_deployment_target=ct.target.iOS18,
        )

        # check the numerical outputs
        state1 = mlmodel.make_state()
        assert mlmodel.predict({"x": np.array([2.0])}, state=state1)["y"] == 4  # (2)^2
        assert mlmodel.predict({"x": np.array([5.0])}, state=state1)["y"] == 49  # (5+2)^2
        assert mlmodel.predict({"x": np.array([-1.0])}, state=state1)["y"] == 36  # (-1+5+2)^2

        state2 = mlmodel.make_state()
        assert mlmodel.predict({"x": np.array([9.0])}, state=state2)["y"] == 81  # (9)^2
        assert mlmodel.predict({"x": np.array([2.0])}, state=state2)["y"] == 121  # (2+9)^2

        assert mlmodel.predict({"x": np.array([3.0])}, state=state1)["y"] == 81  # (3-1+5+2)^2
        assert mlmodel.predict({"x": np.array([7.0])}, state=state1)["y"] == 256  # (7+3-1+5+2)^2

    @staticmethod
    @pytest.mark.skipif(
        ct.utils._macos_version() < (15, 0), reason="States are supported since iOS18/macos15."
    )
    def test_attention_stateful_key_value_cache():
        """
        Use a toy attention model to showcase kv cache with states.

        This toy example is only for showing how to convert in-place update kv-cache. It omits some
        other details such as multi-head, multi-layer, positional encoding, final logits, etc.
        """

        class SimpleAttention(nn.Module):
            def __init__(self, embed_size):
                super().__init__()
                self.query = nn.Linear(embed_size, embed_size)
                self.key = nn.Linear(embed_size, embed_size)
                self.value = nn.Linear(embed_size, embed_size)

            def forward(self, x):
                Q = self.query(x)  # (batch_size, seq_len, embed_size)
                K = self.key(x)  # (batch_size, seq_len, embed_size)
                V = self.value(x)  # (batch_size, seq_len, embed_size)
                return torch.nn.functional.scaled_dot_product_attention(Q, K, V)

        class ToyModel(nn.Module):
            def __init__(self, vocab_size, embed_size):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, embed_size)
                self.attention = SimpleAttention(embed_size)
                self.fc = nn.Linear(embed_size, embed_size)

            def forward(self, x):
                embedded = self.embedding(x)
                attention_output = self.attention(embedded)
                return self.fc(attention_output)

        class SimpleAttentionWithKeyValueCache(SimpleAttention):
            """Add kv-cache into SimpleAttention."""

            def forward(self, x, attention_mask, k_cache, v_cache):
                Q = self.query(x)
                newly_computed_k = self.key(x)
                newly_computed_v = self.value(x)

                # Update kv-cache in-place.
                q_len = Q.shape[-2]
                end_step = attention_mask.shape[-1]
                past_kv_len = end_step - q_len
                k_cache[:, past_kv_len:end_step, :] = newly_computed_k
                v_cache[:, past_kv_len:end_step, :] = newly_computed_v

                # The K and V we need is (batch_size, q_len + past_kv_len, embed_size).
                K = k_cache[:, :end_step, :]
                V = v_cache[:, :end_step, :]

                return torch.nn.functional.scaled_dot_product_attention(
                    Q, K, V, attn_mask=attention_mask
                )

        class ToyModelWithKeyValueCache(nn.Module):
            def __init__(self, vocab_size, embed_size, batch_size, max_seq_len):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, embed_size)
                self.attention = SimpleAttentionWithKeyValueCache(embed_size)
                self.fc = nn.Linear(embed_size, embed_size)

                self.kvcache_shape = (batch_size, max_seq_len, embed_size)
                self.register_buffer("k_cache", torch.zeros(self.kvcache_shape))
                self.register_buffer("v_cache", torch.zeros(self.kvcache_shape))

            def forward(
                self,
                input_ids,  # [batch_size, seq_len]
                causal_mask,  # [batch_size, seq_len, seq_len + past_kv_len]
            ):
                embedded = self.embedding(input_ids)
                attention_output = self.attention(embedded, causal_mask, self.k_cache, self.v_cache)
                return self.fc(attention_output)

        # If you want to compare prediction speed, the benefits of stateful kv-cache will only be
        # revealed with large models, such as `vocab_size=32000` and `embed_size = 1024`.
        vocab_size = 100
        embed_size = 32
        batch_size = 1
        seq_len = 5
        max_seq_len = 1024
        num_iterations = 100

        # Stateless model without kv-cache.
        torch_model = ToyModel(vocab_size, embed_size)
        torch_model.eval()
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        torch_output = torch_model(input_ids).detach().numpy()
        traced_model = torch.jit.trace(torch_model, [input_ids])
        query_length = ct.RangeDim(lower_bound=1, upper_bound=max_seq_len, default=1)
        inputs = [ct.TensorType(shape=(batch_size, query_length), dtype=np.int32, name="input_ids")]
        outputs = [ct.TensorType(dtype=np.float16, name="output")]

        # The minimum_deployment_target and compute_units is not necessary, as non-stateful models
        # are supported before iOS18. Here we set it just for fair comparison with the stateful
        # kvcache model below.
        converted_model = ct.convert(
            traced_model,
            inputs=inputs,
            outputs=outputs,
            minimum_deployment_target=ct.target.iOS18,
            compute_units=ct.ComputeUnit.CPU_AND_GPU,
        )

        # Makes sure prediction works well.
        for token_id in range(0, num_iterations):
            inputs = {"input_ids": np.array([list(range(token_id + 1))], dtype=np.int32)}
            converted_model.predict(inputs)

        # Stateful model with kv-cache.
        past_kv_len = 0
        torch_model_kvcache = ToyModelWithKeyValueCache(
            vocab_size, embed_size, batch_size, max_seq_len
        )
        torch_model_kvcache.load_state_dict(torch_model.state_dict(), strict=False)
        torch_model_kvcache.eval()
        causal_mask = torch.zeros((batch_size, seq_len, seq_len + past_kv_len), dtype=torch.float32)

        # Make sure the output matches the non-kv-cache version.
        torch_kvcache_output = torch_model_kvcache(input_ids, causal_mask).detach().numpy()
        np.testing.assert_allclose(torch_output, torch_kvcache_output)

        traced_model_kvcache = torch.jit.trace(torch_model_kvcache, [input_ids, causal_mask])
        query_length = ct.RangeDim(lower_bound=1, upper_bound=max_seq_len, default=1)
        end_step_dim = ct.RangeDim(lower_bound=1, upper_bound=max_seq_len, default=1)
        inputs = [
            ct.TensorType(shape=(batch_size, query_length), dtype=np.int32, name="input_ids"),
            ct.TensorType(
                shape=(batch_size, query_length, end_step_dim), dtype=np.float16, name="causal_mask"
            ),
        ]
        outputs = [ct.TensorType(dtype=np.float16, name="output")]

        # In addition to `inputs` and `outputs`, we need `states` which uses the same name as the
        # registered buffers in `ToyModelWithKeyValueCache`.
        states = [
            ct.StateType(
                wrapped_type=ct.TensorType(
                    shape=torch_model_kvcache.kvcache_shape, dtype=np.float16
                ),
                name="k_cache",
            ),
            ct.StateType(
                wrapped_type=ct.TensorType(
                    shape=torch_model_kvcache.kvcache_shape, dtype=np.float16
                ),
                name="v_cache",
            ),
        ]
        converted_model_kvcache = ct.convert(
            traced_model_kvcache,
            inputs=inputs,
            outputs=outputs,
            states=states,
            minimum_deployment_target=ct.target.iOS18,
            compute_units=ct.ComputeUnit.CPU_AND_GPU,
        )

        # Makes sure prediction works well.
        past_kv_len = 0
        kv_cache_state = converted_model_kvcache.make_state()
        for token_id in range(0, num_iterations):
            inputs = {
                "input_ids": np.array([[token_id]], dtype=np.int32),
                "causal_mask": np.zeros((1, 1, past_kv_len + 1), dtype=np.float16),
            }
            converted_model_kvcache.predict(inputs, kv_cache_state)
            past_kv_len += 1


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
                # invocation usually have zero vectors as initial states.
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
            return x + 5.0
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
        with pytest.raises(
            ValueError,
            match="Number of TorchScript inputs \(2\) must match the user provided inputs \(1\).",
        ):
            ct.convert(
                float32_two_input_model,
                inputs=[ct.TensorType(shape=(10, 20), dtype=np.int32)],
                minimum_deployment_target=ct.target.macOS12,
            )

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

        # check mlprogram can have dynamic shape image output
        shape = ct.Shape((1, 3, ct.RangeDim(5, 10), ct.RangeDim(5, 10)))
        mlmodel = ct.convert(
            rank4_input_model,
            inputs=[ct.TensorType(shape=shape, dtype=np.float32)],
            outputs=[ct.ImageType(name="output_image", color_layout=ct.colorlayout.RGB)],
            minimum_deployment_target=ct.target.macOS13,
        )
        assert_ops_in_mil_program(mlmodel, expected_op_list=["cast", "add", "cast"])
        assert_spec_output_image_type(mlmodel._spec, expected_feature_type=ft.ImageFeatureType.RGB)
        assert_prog_input_type(mlmodel._mil_program, expected_dtype_str="fp32")
        assert_prog_output_type(mlmodel._mil_program, expected_dtype_str="fp32")
        assert any_symbolic(mlmodel._mil_program.functions["main"].outputs[0].shape)
        verify_prediction(mlmodel)

        # Test output image numerical
        sample_input = np.random.randint(low=0, high=200, size=(1, 3, 10, 10)).astype(np.float32)
        model_output_pil_image = mlmodel.predict({"x": sample_input})["output_image"]
        assert isinstance(model_output_pil_image, Image.Image)
        assert model_output_pil_image.mode == "RGBA"
        model_output_as_numpy = np.array(model_output_pil_image)[:, :, :3]  # last A channel is 255
        model_output_as_numpy = np.transpose(model_output_as_numpy, axes=[2, 0, 1])
        reference_output = rank4_input_model(torch.from_numpy(sample_input)).detach().numpy()
        reference_output = np.squeeze(reference_output)
        np.testing.assert_allclose(reference_output, model_output_as_numpy, rtol=1e-2, atol=1e-2)

        a_channel = np.array(model_output_pil_image)[:, :, 3].flatten()
        assert np.all(a_channel == 255)

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

    @pytest.mark.parametrize(
        "dynamic_shape",
        [True, False],
    )
    def test_grayscale_output_image(self, rank4_grayscale_input_model, dynamic_shape):

        if dynamic_shape:
            shape = ct.Shape((1, 1, ct.RangeDim(5, 10), ct.RangeDim(5, 20)))
        else:
            shape = (1, 1, 10, 20)

        mlmodel = ct.convert(
            rank4_grayscale_input_model,
            inputs=[ct.TensorType(name="input", shape=shape)],
            outputs=[ct.ImageType(name="output_image", color_layout=ct.colorlayout.GRAYSCALE)],
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

    @pytest.mark.parametrize(
        "dynamic_shape",
        [True, False],
    )
    def test_grayscale_fp16_output_image(self, rank4_grayscale_input_model, dynamic_shape):

        if dynamic_shape:
            shape = ct.Shape((1, 1, ct.RangeDim(5, 10), ct.RangeDim(5, 20)))
        else:
            shape = (1, 1, 10, 20)

        mlmodel = ct.convert(
            rank4_grayscale_input_model,
            inputs=[ct.TensorType(name="input", shape=shape)],
            outputs=[
                ct.ImageType(name="output_image", color_layout=ct.colorlayout.GRAYSCALE_FLOAT16)
            ],
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


@pytest.mark.skipif(
    Version(torch.__version__) < Version("2.4.0"),
    reason="Most torchao functionalities only work with PyTorch 2.4.0+",
)
@pytest.mark.skipif(
    ct.utils._macos_version() < (15, 0),
    reason="Torchao block-wise quantization requires MacOS 15+.",
)
@pytest.mark.skipif(not _HAS_TORCHAO, reason=MSG_TORCHAO_NOT_FOUND)
class TestTorchao:
    """
    This class tests the torchao quantized model conversion.
    """

    @staticmethod
    def _construct_test_model():
        # The old Quantizer method in torchao doesn't work with a single-layer model such as model=nn.Linear(...),
        # so we have to create a Module which contains linear layers.
        class TestModel(nn.Module):
            def __init__(self):
                super(TestModel, self).__init__()
                # Currently torchao only supports Linear module without bias.
                self.linear1 = nn.Linear(32, 64, bias=False)
                self.linear2 = nn.Linear(64, 32, bias=False)
                self.relu = nn.ReLU()

            def forward(self, x):
                x = self.relu(self.linear1(x))
                return self.relu(self.linear2(x))

        return TestModel().to(torch.device("cpu")).eval()

    @pytest.mark.parametrize("use_export", (False, True))
    def test_weight_only_quantization(self, use_export):
        model = self._construct_test_model()
        quantizer = quant_api.Int4WeightOnlyQuantizer(
            precision=torch.float32, groupsize=32, inner_k_tiles=2, device=torch.device("cpu")
        )
        model = quantizer.quantize(model)
        input_data = torch.randn((2, 32), dtype=torch.float16)

        if use_export:
            exported_model = torch.export.export(model, (input_data,))
            inputs = None
        else:
            exported_model = torch.jit.trace(model, example_inputs=(input_data,))
            inputs = [ct.TensorType(shape=input_data.shape, name="input")]

        converted_model = ct.convert(
            exported_model, inputs=inputs, minimum_deployment_target=ct.target.iOS18
        )
        main_func = converted_model._mil_program.functions["main"]
        quantize_ops = main_func.find_ops(op_type="constexpr_blockwise_shift_scale")
        assert len(quantize_ops) > 0

        if ct.utils._is_macos():
            result = converted_model.predict(
                {
                    list(converted_model.input_description)[0]: input_data.detach()
                    .numpy()
                    .astype(np.float32)
                }
            )
            expected = model(input_data)
            output_name = list(result.keys())[0]
            np.testing.assert_allclose(result[output_name], expected.detach().numpy(), atol=1e-3)

    def test_weight_only_quantization_bfloat16_not_support(self):
        """
        Torchao quant_api.int4_weight_only only supports bfloat16.
        """
        model = self._construct_test_model().bfloat16()
        quant_api.quantize_(model, quant_api.int4_weight_only(group_size=32, inner_k_tiles=2))
        model = unwrap_tensor_subclass(model)
        input_data = torch.randn((2, 32), dtype=torch.float16)
        exported_model = torch.export.export(model, (input_data,))
        # The conversion of bfloat16 hasn't been supported yet.
        with pytest.raises(KeyError, match="torch.bfloat16"):
            ct.convert(exported_model, minimum_deployment_target=ct.target.iOS17)

    @pytest.mark.parametrize("use_export", (True, False))
    def test_dynamic_activation_quantization_not_support(self, use_export):
        """
        Although Int8DynActInt4WeightQuantizer will be deprecated, we still want
        to test it because it's used in ExecuTorch to quantize llama models.
        """
        model = self._construct_test_model()
        quantizer = quant_api.Int8DynActInt4WeightQuantizer(
            precision=torch.float16, groupsize=32, device=torch.device("cpu")
        )
        model = quantizer.quantize(model)
        input_data = torch.randn((2, 32), dtype=torch.float16)

        if use_export:
            exported_model = torch.export.export(model, (input_data,))
            inputs = None
            err_msg = "Unsupported fx node quantize_per_token"
            err_type = ValueError
        else:
            exported_model = torch.jit.trace(model, example_inputs=(input_data,))
            inputs = [ct.TensorType(shape=input_data.shape)]
            err_msg = "Dynamic activation quantization is not supported in Core ML"
            err_type = NotImplementedError

        with pytest.raises(err_type, match=err_msg):
            ct.convert(exported_model, inputs=inputs, minimum_deployment_target=ct.target.iOS17)


class TestUtilsImport:
    @staticmethod
    def test_import_construct_matmul():
        """
        _construct_matmul is an utility function that used by some 3rd party codes,
        so here we make sure that this method is exposed.
        """
        from coremltools.converters.mil.frontend.torch.ops import _construct_matmul

        assert _construct_matmul is not None
