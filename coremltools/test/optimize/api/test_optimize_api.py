#  Copyright (c) 2023, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import tempfile

import pytest

from coremltools.converters.mil.testing_utils import get_op_types_in_program
from coremltools.test.optimize.coreml.test_passes import (
    TestCompressionPasses as _TestCompressionPasses,
)
from coremltools.test.optimize.coreml.test_passes import (
    TestConfigurationFromDictFromYaml as _TestConfigurationFromDictFromYaml,
)

get_test_program = _TestCompressionPasses._get_test_program_2


def create_model_and_optimizer():
    import torch

    class Model(torch.nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.conv1 = torch.nn.Conv2d(3, 128, (1, 1))
            self.conv2 = torch.nn.Conv2d(128, 256, (10, 10))
            self.conv3 = torch.nn.Conv2d(256, 26, (10, 10))
            self.linear = torch.nn.Linear(206, 12)

        def forward(self, x):
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.linear(x)
            return x

    model = Model()
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    return model, loss_fn, optimizer


def get_mlmodel():
    import coremltools as ct
    prog = get_test_program()
    mlmodel = ct.convert(prog, convert_to="mlprogram", compute_precision=ct.precision.FLOAT32)
    return mlmodel


class TestOptimizeCoremlAPIOverview:
    """
    This class is testing the api reference code in
    https://coremltools.readme.io/v7.0/docs/optimizecoreml-api-overview
    """

    def test_6_bit_palettization_example(self):
        import coremltools as ct
        import coremltools.optimize.coreml as cto

        # load model
        # (original) mlmodel = ct.models.MLModel(uncompressed_model_path)
        mlmodel = get_mlmodel()

        # define op config
        op_config = cto.OpPalettizerConfig(mode="kmeans", nbits=6)

        # define optimization config by applying the op config globally to all ops
        config = cto.OptimizationConfig(global_config=op_config)

        # palettize weights
        compressed_mlmodel = cto.palettize_weights(mlmodel, config)

        # Do some basic checks
        assert compressed_mlmodel is not None
        ops = get_op_types_in_program(compressed_mlmodel._mil_program)
        assert ops.count("constexpr_lut_to_dense") == 6

    def test_linear_quantization_config_from_yaml(self):
        import coremltools.optimize.coreml as cto

        mlmodel = get_mlmodel()

        config_dict = {
            "config_type": "OpLinearQuantizerConfig",
            "global_config": {
                "mode": "linear_symmetric",
                "dtype": "int8",
            },
        }
        yaml_file = _TestConfigurationFromDictFromYaml.get_yaml(config_dict)

        # (original) config = cto.OptimizationConfig.from_yaml("linear_config.yaml")
        config = cto.OptimizationConfig.from_yaml(yaml_file)
        compressed_mlmodel = cto.linear_quantize_weights(mlmodel, config)

        # Do some basic checks
        assert compressed_mlmodel is not None
        ops = get_op_types_in_program(compressed_mlmodel._mil_program)
        assert ops.count("constexpr_affine_dequantize") == 6

    def test_customize_ops_to_compress(self):
        import coremltools.optimize.coreml as cto

        mlmodel = get_mlmodel()

        global_config = cto.OpPalettizerConfig(nbits=6, mode="kmeans")
        linear_config = cto.OpPalettizerConfig(nbits=8, mode="kmeans")
        config = cto.OptimizationConfig(
            global_config=global_config,
            op_type_configs={"linear": linear_config},
            op_name_configs={"conv1": None, "conv3": None},
        )
        compressed_mlmodel = cto.palettize_weights(mlmodel, config)

        # Do some basic checks
        assert compressed_mlmodel is not None
        ops = get_op_types_in_program(compressed_mlmodel._mil_program)
        assert ops.count("constexpr_lut_to_dense") == 4


class TestOptimizeTorchAPIOverview:
    """
    This class is testing the api reference code in
    https://coremltools.readme.io/v7.0/docs/optimizetorch-api-overview
    """

    def get_global_config(self):
        config_dict = {
            "global_config": {
                "scheduler": {"update_steps": [100, 200, 300, 500]},
                "target_sparsity": 0.8,
            }
        }
        return _TestConfigurationFromDictFromYaml.get_yaml(config_dict)

    def get_fine_grain_config(self):
        config_dict = {
            "module_type_configs": {
                "Linear": {
                    "scheduler": {
                        "update_steps": [100, 200, 300, 500],
                    },
                    "n_m_ratio": [3, 4],
                },
                "Conv2d": {
                    "scheduler": {
                        "update_steps": [100, 200, 300, 500],
                    },
                    "target_sparsity": 0.5,
                    "block_size": 2,
                },
            },
            "module_name_configs": {
                "module2.conv1": {
                    "scheduler": {
                        "update_steps": [100, 200, 300, 500],
                    },
                    "target_sparsity": 0.75,
                },
                "module2.linear": None,
            },
        }
        return _TestConfigurationFromDictFromYaml.get_yaml(config_dict)

    def test_load_from_yaml(self):
        def _test_config(config_path):
            import torch

            import coremltools as ct
            from coremltools.optimize.torch.pruning import MagnitudePruner, MagnitudePrunerConfig

            # Toy example
            x, label = torch.rand(1, 3, 224, 224), torch.rand(1, 26, 206, 12)
            data = [(x, label)]

            model, loss_fn, optimizer = create_model_and_optimizer()

            # Initialize pruner and configure it
            # (original) config = MagnitudePrunerConfig.from_yaml("config.yaml")
            config = MagnitudePrunerConfig.from_yaml(config_path)

            pruner = MagnitudePruner(model, config)

            # Insert pruning layers in the model
            model = pruner.prepare()

            for inputs, labels in data:
                output = model(inputs)
                loss = loss_fn(output, labels)
                loss.backward()
                optimizer.step()
                pruner.step()

            # Commit pruning masks to model parameters
            pruner.finalize(inplace=True)

            # Export
            example_input = torch.rand(1, 3, 224, 224)
            traced_model = torch.jit.trace(model, example_input)

            coreml_model = ct.convert(
                traced_model,
                inputs=[ct.TensorType(shape=example_input.shape)],
                pass_pipeline=ct.PassPipeline.DEFAULT_PRUNING,
                minimum_deployment_target=ct.target.iOS16,
            )
            assert coreml_model is not None
            output_file = tempfile.NamedTemporaryFile(suffix=".mlpackage").name
            coreml_model.save(output_file)

        _test_config(self.get_global_config())
        _test_config(self.get_fine_grain_config())

    @pytest.mark.xfail(reason="rdar://132361333 Palettization Test Case Time Out", run=False)
    def test_programmatic_example_1(self):
        import torch

        import coremltools as ct
        from coremltools.optimize.torch.palettization import (
            DKMPalettizer,
            DKMPalettizerConfig,
            ModuleDKMPalettizerConfig,
        )

        # Toy example
        x, label = torch.rand(1, 3, 224, 224), torch.rand(1, 26, 206, 12)
        data = [(x, label)]

        # code that defines the pytorch model, and optimizer
        model, loss_fn, optimizer = create_model_and_optimizer()

        # Initialize the palettizer
        config = DKMPalettizerConfig(
            global_config=ModuleDKMPalettizerConfig(n_bits=4, cluster_dim=2)
        )

        palettizer = DKMPalettizer(model, config)

        # Prepare the model to insert FakePalettize layers for palettization
        model = palettizer.prepare(inplace=True)

        # Use palettizer in the PyTorch training loop
        for inputs, labels in data:
            output = model(inputs)
            loss = loss_fn(output, labels)
            loss.backward()
            optimizer.step()
            palettizer.step()

        # Fold LUT and indices into weights
        model = palettizer.finalize(inplace=True)

        # Export
        example_input = torch.rand(1, 3, 224, 224)
        traced_model = torch.jit.trace(model, example_input)

        coreml_model = ct.convert(
            traced_model,
            inputs=[ct.TensorType(shape=example_input.shape)],
            pass_pipeline=ct.PassPipeline.DEFAULT_PALETTIZATION,
            minimum_deployment_target=ct.target.iOS18,
        )
        assert coreml_model is not None
        output_file = tempfile.NamedTemporaryFile(suffix=".mlpackage").name
        coreml_model.save(output_file)

    def test_programmatic_example_2(self):
        import torch

        import coremltools as ct
        from coremltools.optimize.torch.quantization import (
            LinearQuantizer,
            LinearQuantizerConfig,
            ModuleLinearQuantizerConfig,
            ObserverType,
            QuantizationScheme,
        )

        # Toy example
        x, label = torch.rand(1, 3, 224, 224), torch.rand(1, 26, 206, 12)
        data = [(x, label)]
        model, loss_fn, optimizer = create_model_and_optimizer()

        # Initialize the quantizer
        global_config = ModuleLinearQuantizerConfig(
            quantization_scheme=QuantizationScheme.symmetric
        )

        config = LinearQuantizerConfig().set_global(global_config)

        # We only want to quantize convolution layers which have a kernel size of 1 or all linear layers.
        for name, m in model.named_modules():
            if isinstance(m, torch.nn.Conv2d):
                if m.kernel_size == (1, 1):
                    config = config.set_module_name(
                        name,
                        ModuleLinearQuantizerConfig(
                            weight_observer=ObserverType.min_max, weight_per_channel=True
                        ),
                    )
                else:
                    config = config.set_module_name(name, None)

        quantizer = LinearQuantizer(model, config)

        # Prepare the model to insert FakeQuantize layers for QAT
        example_input = torch.rand(1, 3, 224, 224)
        model = quantizer.prepare(example_inputs=example_input, inplace=True)

        # Use quantizer in your PyTorch training loop
        for inputs, labels in data:
            output = model(inputs)
            loss = loss_fn(output, labels)
            loss.backward()
            optimizer.step()
            quantizer.step()

        # Convert operations to their quanitzed counterparts using parameters learnt via QAT
        model = quantizer.finalize(inplace=True)

        traced_model = torch.jit.trace(model, example_input)

        coreml_model = ct.convert(
            traced_model,
            inputs=[ct.TensorType(shape=example_input.shape)],
            minimum_deployment_target=ct.target.iOS17,
        )
        assert coreml_model is not None
        output_file = tempfile.NamedTemporaryFile(suffix=".mlpackage").name
        coreml_model.save(output_file)

    def test_quantize_submodule(self):
        import torch
        from torchvision.models import mobilenet_v3_small

        import coremltools as ct
        from coremltools.optimize.torch.quantization import LinearQuantizer

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model1 = mobilenet_v3_small()
                self.model2 = mobilenet_v3_small()

            def forward(self, x):
                return self.model1(x), self.model2(x)

        model = Model()
        data = torch.randn(1, 3, 224, 224)
        example_inputs = (data,)

        quantizer = LinearQuantizer(model.model1)
        model.model1 = quantizer.prepare(example_inputs=example_inputs)
        model(data)
        model.model1 = quantizer.finalize()

        model = model.eval()
        traced_model = torch.jit.trace(model, example_inputs=example_inputs)
        coreml_model = ct.convert(
            traced_model,
            convert_to="mlprogram",
            inputs=[ct.TensorType(shape=data.shape)],
            minimum_deployment_target=ct.target.iOS18,
            skip_model_load=True,
        )
        assert coreml_model is not None
        quant_ops = coreml_model._mil_program.functions["main"].find_ops(
            op_type="constexpr_blockwise_shift_scale"
        )
        assert len(quant_ops) > 0


class TestConvertingCompressedSourceModels:
    """
    This class is testing examples in https://coremltools.readme.io/v7.0/docs/converting-compressed-source-models
    """

    def test_smoke_convert_compressed_source_model_pruning(self):
        import coremltools as ct

        model_with_sparse_weights = ct.convert(
            get_test_program(),
            pass_pipeline=ct.PassPipeline.DEFAULT_PRUNING,
            minimum_deployment_target=ct.target.iOS17,
        )
        assert model_with_sparse_weights is not None

    def test_smoke_convert_compressed_source_model_pelettization(self):
        import coremltools as ct

        model_with_lut_weights = ct.convert(
            get_test_program(),
            pass_pipeline=ct.PassPipeline.DEFAULT_PALETTIZATION,
            minimum_deployment_target=ct.target.macOS13,
        )
        assert model_with_lut_weights is not None


class TestPostTrainingPruning:
    """
    This class is testing examples in https://coremltools.readme.io/v7.0/docs/pruning-a-core-ml-model
    """

    def test_threshold_pruner(self):
        from coremltools.optimize.coreml import (
            OpThresholdPrunerConfig,
            OptimizationConfig,
            prune_weights,
        )

        model = get_mlmodel()
        op_config = OpThresholdPrunerConfig(
            threshold=0.03,
            minimum_sparsity_percentile=0.55,
            weight_threshold=1024,
        )
        config = OptimizationConfig(global_config=op_config)
        model_compressed = prune_weights(model, config=config)
        assert model_compressed is not None

    def test_magnitute_pruner(self):
        from coremltools.optimize.coreml import (
            OpMagnitudePrunerConfig,
            OptimizationConfig,
            prune_weights,
        )

        model = get_mlmodel()
        op_config = OpMagnitudePrunerConfig(
            target_sparsity=0.6,
            weight_threshold=1024,
        )
        config = OptimizationConfig(global_config=op_config)
        model_compressed = prune_weights(model, config=config)


class TestTrainingTimePruning:
    """
    This class is testing examples in https://coremltools.readme.io/v7.0/docs/data-dependent-pruning
    """

    def test_magnitute_pruner(self):
        from collections import OrderedDict

        import torch

        import coremltools as ct
        from coremltools.optimize.torch.pruning import MagnitudePruner, MagnitudePrunerConfig

        # Toy example
        x, label = torch.rand(1, 3, 224, 224), torch.rand(1, 32, 224, 224)
        data = [(x, label)]
        model = torch.nn.Sequential(
            OrderedDict(
                [
                    ("conv1", torch.nn.Conv2d(3, 32, 3, padding="same")),
                    ("conv2", torch.nn.Conv2d(32, 32, 3, padding="same")),
                ]
            )
        )
        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

        # initialize pruner and configure it
        # we will configure the pruner for all conv2d layers
        config = MagnitudePrunerConfig.from_dict(
            {
                "module_type_configs": {
                    "Conv2d": {
                        "scheduler": {"update_steps": [3, 5, 7]},
                        "target_sparsity": 0.75,
                        "granularity": "per_scalar",
                    },
                }
            }
        )

        pruner = MagnitudePruner(model, config)

        # insert pruning layers in the model
        model = pruner.prepare()

        for inputs, labels in data:
            output = model(inputs)
            loss = loss_fn(output, labels)
            loss.backward()
            optimizer.step()
            pruner.step()

        # commit pruning masks to model parameters
        pruner.finalize(inplace=True)

        # trace and convert the model
        example_input = torch.rand(1, 3, 224, 224)  # shape of input for the model
        traced_model = torch.jit.trace(model, example_input)
        coreml_model = ct.convert(
            traced_model,
            convert_to="mlprogram",
            inputs=[ct.TensorType(shape=example_input.shape)],
            pass_pipeline=ct.PassPipeline.DEFAULT_PRUNING,
            minimum_deployment_target=ct.target.iOS17,
        )

        assert coreml_model is not None
        output_file = tempfile.NamedTemporaryFile(suffix=".mlpackage").name
        coreml_model.save(output_file)


class TestPostTrainingPalettization:
    """
    This class is testing the examples in https://coremltools.readme.io/v7.0/docs/data-free-palettization
    """

    def test_palettizer(self):
        from coremltools.optimize.coreml import (
            OpPalettizerConfig,
            OptimizationConfig,
            palettize_weights,
        )

        model = get_mlmodel()
        op_config = OpPalettizerConfig(mode="kmeans", nbits=6, weight_threshold=512)
        config = OptimizationConfig(global_config=op_config)
        compressed_6_bit_model = palettize_weights(model, config=config)

        # Some basic checks
        assert compressed_6_bit_model is not None
        ops = get_op_types_in_program(compressed_6_bit_model._mil_program)
        assert ops.count("constexpr_lut_to_dense") == 6


class TestTrainingTimePalettization:
    """
    This class is testing the examples in https://coremltools.readme.io/v7.0/docs/data-dependent-palettization
    """

    def test_palettizer(self):
        import torch
        import torch.nn as nn

        import coremltools as ct
        from coremltools.optimize.torch.palettization import DKMPalettizer, DKMPalettizerConfig

        # Toy example
        x, label = torch.rand(1, 4), torch.rand(1, 4)
        data = [(x, label)]

        model = nn.Sequential(nn.Linear(4, 500), nn.Sigmoid(), nn.Linear(500, 4), nn.Sigmoid())
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

        # Prepare model for palettization
        module_config = {nn.Linear: {"n_bits": 2, "weight_threshold": 1000, "milestone": 2}}
        config = DKMPalettizerConfig.from_dict({"module_type_configs": module_config})
        palettizer = DKMPalettizer(model, config)

        prepared_model = palettizer.prepare()

        # Fine-tune the model for a few epochs after this.
        for inputs, labels in data:
            output = model(inputs)
            loss = loss_fn(output, labels)
            loss.backward()
            optimizer.step()
            palettizer.step()

        # prepare for conversion
        finalized_model = palettizer.finalize()

        # trace and convert
        example_input = torch.rand(1, 4)  # shape of input for the model
        traced_model = torch.jit.trace(finalized_model, example_input)

        coreml_model = ct.convert(
            traced_model,
            convert_to="mlprogram",
            inputs=[ct.TensorType(shape=example_input.shape)],
            pass_pipeline=ct.PassPipeline.DEFAULT_PALETTIZATION,
            minimum_deployment_target=ct.target.iOS16,
        )

        assert coreml_model is not None
        output_file = tempfile.NamedTemporaryFile(suffix=".mlpackage").name
        coreml_model.save(output_file)


class TestPostTrainingQuantization:
    """
    This class is testing the examples in https://coremltools.readme.io/v7.0/docs/data-free-quantization
    """

    def test_quantization(self):
        import coremltools.optimize.coreml as cto

        model = get_mlmodel()
        op_config = cto.OpLinearQuantizerConfig(mode="linear_symmetric", weight_threshold=512)
        config = cto.OptimizationConfig(global_config=op_config)

        compressed_8_bit_model = cto.linear_quantize_weights(model, config=config)

        # Some basic checks
        assert compressed_8_bit_model is not None
        ops = get_op_types_in_program(compressed_8_bit_model._mil_program)
        assert ops.count("constexpr_affine_dequantize") == 6


class TestTrainingTimeQuantization:
    """
    This class is testing the examples in https://coremltools.readme.io/v7.0/docs/data-dependent-quantization
    """

    def test_quantization(self):
        from collections import OrderedDict

        import torch
        import torch.nn as nn

        import coremltools as ct
        from coremltools.optimize.torch.quantization import LinearQuantizer, LinearQuantizerConfig

        # Toy example
        x, label = torch.rand(1, 1, 20, 20), torch.rand(1, 20, 16, 16)
        data = [(x, label)]

        model = nn.Sequential(
            OrderedDict(
                {
                    "conv": nn.Conv2d(1, 20, (3, 3)),
                    "relu1": nn.ReLU(),
                    "conv2": nn.Conv2d(20, 20, (3, 3)),
                    "relu2": nn.ReLU(),
                }
            )
        )

        loss_fn = nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

        # Initialize the quantizer
        config = LinearQuantizerConfig.from_dict(
            {
                "global_config": {
                    "quantization_scheme": "symmetric",
                    "milestones": [0, 100, 400, 200],
                }
            }
        )
        quantizer = LinearQuantizer(model, config)

        # Prepare the model to insert FakeQuantize layers for QAT
        example_input = torch.rand(1, 1, 20, 20)
        model = quantizer.prepare(example_inputs=example_input, inplace=True)

        # Use quantizer in your PyTorch training loop
        for inputs, labels in data:
            output = model(inputs)
            loss = loss_fn(output, labels)
            loss.backward()
            optimizer.step()
            quantizer.step()

        # Convert operations to their quanitzed counterparts using parameters learnt via QAT
        model = quantizer.finalize(inplace=True)

        # Convert the PyTorch models to CoreML format
        traced_model = torch.jit.trace(model, example_input)
        coreml_model = ct.convert(
            traced_model,
            convert_to="mlprogram",
            inputs=[ct.TensorType(shape=example_input.shape)],
            minimum_deployment_target=ct.target.iOS17,
        )

        assert coreml_model is not None
        output_file = tempfile.NamedTemporaryFile(suffix=".mlpackage").name
        coreml_model.save(output_file)
