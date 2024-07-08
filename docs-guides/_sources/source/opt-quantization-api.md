API Overview
=============

## Working with Core ML Models

### Quantizing weights

You can linearly quantize the weights of your Core ML model by using the 
[``linear_quantize_weights``](https://apple.github.io/coremltools/source/coremltools.optimize.coreml.post_training_quantization.html#coremltools.optimize.coreml.linear_quantize_weights) method as follows:

```python
import coremltools.optimize as cto

op_config = cto.coreml.OpLinearQuantizerConfig(
    mode="linear_symmetric", weight_threshold=512
)
config = cto.coreml.OptimizationConfig(global_config=op_config)

compressed_8_bit_model = cto.coreml.linear_quantize_weights(model, config=config)
```

The method defaults to ``linear_symmetric``, which uses only per-channel scales and no zero-points.  
You can also choose a ``linear`` mode which uses a zero-point as well, which may help to get 
slightly better accuracy.

For more details on the parameters available in the config, see the following in the API Reference:

- [``OpLinearQuantizerConfig``](https://apple.github.io/coremltools/source/coremltools.optimize.coreml.config.html#coremltools.optimize.coreml.OpLinearQuantizerConfig)
- [``OptimizationConfig``](https://apple.github.io/coremltools/source/coremltools.optimize.coreml.config.html#coremltools.optimize.coreml.OptimizationConfig)
- [``linear_quantize_weights``](https://apple.github.io/coremltools/source/coremltools.optimize.coreml.post_training_quantization.html#coremltools.optimize.coreml.linear_quantize_weights)

### Quantizing weights and activations

You can also quantize the activations of the model, in addition to the weights, to benefit from
the ``int8``-``int8`` compute available on the Neural Engine (NE), from iPhone 15 Pro onwards.

```python
activation_config = cto.coreml.OptimizationConfig(
    global_config=cto.coreml.experimental.OpActivationLinearQuantizerConfig(
        mode="linear_symmetric"
    )
)

compressed_model_a8 = cto.coreml.experimental.linear_quantize_activations(
    model, activation_config, sample_data
)
```

After quantizing the activation to 8 bits, you can apply the ``linear_quantize_weights`` API 
specified above, to quantize the weights as well, to get an ``W8A8`` model.

## Working with PyTorch Models

### Quantizing weights

#### Data free quantization

To quantize the weights in a data free manner, use 
[PostTrainingQuantizer](https://apple.github.io/coremltools/source/coremltools.optimize.torch.quantization.html#coremltools.optimize.torch.quantization.PostTrainingQuantizer), 
as follows:

```python
import torch
from coremltools.optimize.torch.quantization import PostTrainingQuantizer, \
    PostTrainingQuantizerConfig

config = PostTrainingQuantizerConfig.from_dict(
    {
        "global_config": {
            "weight_dtype": "int8",
            "granularity": "per_block",
            "block_size": 128,
        },
        "module_type_configs": {
            torch.nn.Linear: None
        }
    }
)
quantizer = PostTrainingQuantizer(model, config)
quantized_model = quantizer.compress()
```

- By specifying ``module_type_configs``, one can specify different configs for different layer types. Here, we are setting config for linear layers to be ``None`` to de-select linear layers for quantization. 
- ``granularity`` option allows quantizing the weights at different levels of granularity, like ``per_block``,
where blocks of weights along a channel use same quantization parameters or ``per_channel``, where 
all elements in a channel share the same quantization parameters. Learn more about the various config
options available in
[PostTrainingQuantizerConfig](https://apple.github.io/coremltools/source/coremltools.optimize.torch.quantization.html#coremltools.optimize.torch.quantization.PostTrainingQuantizerConfig).

#### Calibration data based quantization

Use ``LayerwiseComressor`` with ``GPTQ`` algorithm, as follows:

```python
from coremltools.optimize.torch.quantization import LayerwiseCompressor, \
    LayerwiseCompressorConfig

config = LayerwiseCompressorConfig.from_dict(
    {
        "global_config": {
            "algorithm": "gptq",
            "weight_dtype": 4,
            "granularity": "per_block",
            "block_size": 128,
        },
        "input_cacher": "default",
        "calibration_nsamples": 16,
    }
)

dataloader = # create a list of input tensors to be used for calibration

quantizer = LayerwiseCompressor(model, config)

compressed_model = quantizer.compress(dataloader)
```

### Quantizing weights and activations

#### Calibration data based quantization

[``LinearQuantizer``](https://apple.github.io/coremltools/source/coremltools.optimize.torch.quantization.html#coremltools.optimize.torch.quantization.LinearQuantizer),
as described in the next section, is an API to do quantization aware training (QAT) 
for quantizing activations and weights. We can also use the same API for data calibration 
based post training quantization to get a ``W8A8`` model.

We use the calibration data to measure statistics of activations and weights without actually 
simulating quantization during model's forward pass, and without needing to perform a backward pass.
Since the weights are constant and do not change, this amounts to using 
round to nearest (RTN) for quantizing them. 


```python
import torch
from coremltools.optimize.torch.quantization import (
    LinearQuantizer,
    LinearQuantizerConfig,
    ModuleLinearQuantizerConfig
)

config = LinearQuantizerConfig(
    global_config=ModuleLinearQuantizerConfig(
        quantization_scheme="symmetric",
        milestones=[0, 1000, 1000, 0],
    )
)

quantizer = LinearQuantizer(model, config)

quantizer.prepare(example_inputs=[1, 3, 224, 224], inplace=True)

# Only step through quantizer once to enable statistics collection (milestone 0),
# and turn batch norm to inference mode (milestone 3) 
quantizer.step()

# Do a forward pass through the model with calibration data
for idx, data in enumerate(dataloader):
    with torch.no_grad():
        model(data)

model.eval()
quantized_model = quantizer.finalize()
```

Note that, here, we set the first and last values of the ``milestones`` parameter to ``0``. 
The first milestone turns on observes, and setting it to zero ensures that we start measuring 
quantization statistics from step 0. And the last milestone applies batch norm in inference mode,
which means we do not use the calibration data to update the batch norm statistics. We do this because
we do not want training data to influence the batch norm values. The other two milestones 
are used to control when fake quantization simulation is turned on and when observers are turned off.
We can set them to values larger than 0 so that they are never turned on.


#### Quantization Aware Training (QAT)

We use [``LinearQuantizer``](https://apple.github.io/coremltools/source/coremltools.optimize.torch.quantization.html#coremltools.optimize.torch.quantization.LinearQuantizer)
here as well, with a few extra steps, as demonstrated below. 

Specify config in a ``YAML`` file:
```yaml
global_config:
  quantization_scheme: symmetric
  milestones:
    - 0
    - 100
    - 400
    - 200
module_name_configs:
  first_layer: null
  final_layer: null
```

**Code**:
```python
# Initialize the quantizer
config = LinearQuantizerConfig.from_yaml("/path/to/yaml/config.yaml")
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

# Convert operations to their quantized counterparts using parameters learnt via QAT
model = quantizer.finalize(inplace=True)
```

- Here, we have written the configuration as a ``yaml`` file, and used ``module_name_configs`` 
to specify that we do not want first and last layer to be quantized. In actual config, you would 
specify the exact names of the first and last layers to de-select them for quantization. This 
is typically useful, but not required.  
- A detailed explanation of various stages  of quantization can be found in the API Reference for 
[``ModuleLinearQuantizerConfig``](https://apple.github.io/coremltools/source/coremltools.optimize.torch.quantization.html#coremltools.optimize.torch.quantization.ModuleLinearQuantizerConfig).

In QAT, in addition to observing the values of weights and activation tensors
to compute quantization parameters, we also simulate the effects of fake quantization
during training. And instead of just performing forward pass on the model,
we perform full training, with an optimizer. The forward and backward pass computations 
are conducted in ``float32`` dtype. However, these ``float32`` values follow the 
constraints imposed by ``int8`` and ``quint8`` dtypes, for weights and activations respectively. 
This allows the model weights to adjust and reduce the error introduced by quantization. [Straight-Through Estimation](https://arxiv.org/pdf/1308.3432.pdf) 
is used for computing gradients of non-differentiable operations introduced by simulated quantization.

The ``LinearQuantizer`` algorithm is implemented as an extension of 
[FX Graph Mode Quantization](https://pytorch.org/tutorials/prototype/fx_graph_mode_quant_guide.html) in PyTorch. 
It first traces the PyTorch model symbolically to obtain a [torch.fx](https://pytorch.org/docs/stable/fx.html) 
graph capturing all the operations in the model. It then analyzes this graph, 
and inserts [FakeQuantize](https://pytorch.org/docs/stable/generated/torch.ao.quantization.fake_quantize.FakeQuantize.html) layers in the graph. 
FakeQuantize layer insertion locations are chosen such that model inference on hardware is 
optimized and only weights and activations which benefit from quantization are quantized.

Since the prepare method uses [prepare_qat_fx](https://pytorch.org/docs/stable/generated/torch.ao.quantization.quantize_fx.prepare_qat_fx.html) 
to insert quantization layers, the model returned from the method is a 
[torch.fx.GraphModule](https://pytorch.org/docs/stable/fx.html#torch.fx.GraphModule), 
and as a result custom methods defined on the original model class
may not be available on the returned model. Some models, like those with dynamic control 
flow, may not be traceable into a torch.fx.GraphModule. We recommend following the 
instructions in [Limitations of Symbolic Tracing](https://pytorch.org/docs/stable/fx.html#limitations-of-symbolic-tracing) and 
[FX Graph Mode Quantization User Guide](https://pytorch.org/tutorials/prototype/fx_graph_mode_quant_guide.html) to 
update your model first, before using LinearQuantizer algorithm.

### Converting quantized PyTorch models to Core ML

You can convert your PyTorch model, once it has been quantized, as you would a normal PyTorch model:

```python
# Convert the PyTorch models to CoreML format
traced_model = torch.jit.trace(model, example_input)
coreml_model = ct.convert(
    traced_model,
    convert_to="mlprogram",
    inputs=[ct.TensorType(shape=example_input.shape)],
    minimum_deployment_target=ct.target.iOS17,
)
coreml_model.save("~/quantized_model.mlpackage")
```

Note that you need to use ``minimum_deployment_target >= iOS17`` when activations are also quantized. 