Optimizing ResNet50 Model
==========================

In this article we will experiment with various ways to compress a convolutional neural network (CNN) for meeting different performance objectives
while staying within a specified accuracy loss budget.  In particular, we will consider two scenarios - 
one with the goal to reduce the model size, the other with the goal to reduce runtime latency. 

For this exercise, we will use the pretrained [ResNet50](https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html#torchvision.models.resnet50) model from torchvision.
Baseline ResNet50 model has a top 1% accuracy of `76.13%`, mlpackage size of `49MB` (`float16` precision) and latency of `~1.63ms`[^1].

[^1]: All latency numbers reported in this article have been measured on iPhone 15 Pro with iOS18 seed build. The latency numbers are sensitive to the device state, and may vary depending on the device state and build versions.

## Scenario 1 : Minimizing model size

In this scenario, our goal would be to minimize the disk size of the model, while trying to retain as much accuracy as possible of the `float16` model (within 5%).
 
### Palettization using data free compression

Let's start with the quickest workflow, which is to apply data free compression. We will take the model and apply palettization with different bit precisions and see how the accuracy behaves.

```python
from coremltools.optimize.torch.palettization import (
    PostTrainingPalettizer, 
    PostTrainingPalettizerConfig
)

config_dict = {"global_config": {"n_bits": 4}}
config = PostTrainingPalettizerConfig.from_dict(config_dict)
palettizer = PostTrainingPalettizer(model, config)

# Compress model
palettized_model = palettizer.compress()
```

In the above code snippet, we apply data free compression directly to the PyTorch model. 
However, you can also use [ct.optimize.coreml.palettize_weights](https://apple.github.io/coremltools/source/coremltools.optimize.coreml.post_training_quantization.html#coremltools.optimize.coreml.palettize_weights) if working with a Core ML model. 

We apply `n_bits` equal to 8, 6, 4 & 3 bits to get accuracies of `76.09%`, `75.55%`, `66.81%` and `23.09%` respectively. 
We see that there is marginal loss of accuracy with 8 and 6 bit palettized models, whereas there is a big drop with 4 bits, and the model becomes unusable for 3 bits.
                                       
Let's try to regain the accuracy loss with 4 bit palletization by applying the `per_grouped_channel` palettization, which increases the number of LUTs (look up tables) per weight tensor.

```python
from coremltools.optimize.torch.palettization import (
    PostTrainingPalettizer, 
    PostTrainingPalettizerConfig
)

config_dict = {
    "global_config": {
        "n_bits": 4,
        "granularity": "per_grouped_channel",
        "group_size": 8,
    }
}
config = PostTrainingPalettizerConfig.from_dict(config_dict)
palettizer = PostTrainingPalettizer(model, config)

# Compress model
palettized_model = palettizer.compress()
```

We try `group_size` equal to 16, 8 & 4 and see accuracy improve to `69.29%`, `72.26%` and `73.05%` respectively.

We summarize our results below: 

| Config                | Accuracy | Model Size | Latency |
|-----------------------|----------|------------|---------|
| 6-bit (per tensor)    | 75.55%   | 18.6 MB    | 1.25ms  |
| 4-bit (per-tensor)    | 66.81%   | 12.5 MB    | 1.12ms  |
| 4-bit (group_size=16) | 69.29%   | 15.5 MB    | 1.38ms  |
| 4 bit (group_size=8)  | 72.26%   | 12.6 MB    | 1.34ms  |
| 4 bit (group_size=4)  | 73.05%   | 12.7 MB    | 1.71ms  |
 
Note that while higher granularity achieved with grouped channel palettization helps improve accuracy, we may lose runtime performance.  
For this model 4-bit palettization with group_size=8 configuration is a good sweet spot for good accuracy and runtime performance for minimal model size.

### Palettization using fine tuning

For this particular model, we do not see any benefits of using calibration data based compression w.r.to accuracy.
So we move on to training time compression workflow, where we will fine tune the model as we compress it. We can do so by using the [DKM algorithm](opt-palettization-algos.md#differentiable-k-means), as follows:

```python
from coremltools.optimize.torch.palettization import (
    DKMPalettizer,
    DKMPalettizerConfig,
    ModuleDKMPalettizerConfig
)

global_config = ModuleDKMPalettizerConfig(n_bits=2)
config = DKMPalettizerConfig().set_global(global_config)

palettizer = DKMPalettizer(model, config)

palettizer.prepare(inplace=True)

for epoch in range(num_epochs):
    model.train()
    for data, label in enumerate(train_loader):
        train_step(model, optimizer, train_loader, data, label, epoch)
        palettizer.step()

model.eval()
palettized_model = palettizer.finalize()
```

With training time compression we can go up to 1-bit palettized model while still being within our accuracy budget. 
2-bit palettized model has accuracy of `75.51%` and size of `6.3 MB`, while the 1-bit palettized model has accuracy 
of `71.22%` and size of `3.4 MB`, giving over 15x model size reduction over baseline.

### Summary
Below we summarize results from above experiments as well as note the order of time taken to apply each of these compression workflows. 
Post training algorithms are easier to set up and take less time, while training time techniques can provide better 
accuracy to compression trade-off. Note also that while the goal here was to reduce the model size, compressing the model also helps reduce latency (results may vary based on model).

| Optimization API       | Best config          | Accuracy | Model Size | Latency | Time to compress      |
|------------------------|----------------------|----------|------------|---------|-----------------------|
| Baseline               | -                    | 76.13%   | 48.8 MB    | 1.63ms  | -                     |
| PostTrainingPalettizer | 4-bit (group_size=8) | 72.26%   | 13.1 MB    | 1.34ms  | O(minutes)            |
| DKMPalettizer          | 1-bit (per-tensor)   | 71.22%   | 3.4 MB     | 1.14ms  | O(hours) (300 epochs) |


## Scenario 2: Minimizing latency

Next, we will try to minimize latency of our model with less than 5% accuracy loss. Let's say our latency target is < 1ms. 

### Latency reduction with pruning
We start out by pruning the model using data free compression, to see which sparsity configuration gives us the desired latency. 
         
```python
from coremltools.optimize.torch.pruning import (
    MagnitudePruner,
    MagnitudePrunerConfig,
    ModuleMagnitudePrunerConfig
)

global_config = ModuleMagnitudePrunerConfig(initial_sparsity=0.5, target_sparsity=0.5)
config = MagnitudePrunerConfig().set_global(global_config)

pruner = MagnitudePruner(model, config)
pruner.prepare(inplace=True)

# Skip training 

pruned_model = pruner.finalize()
```

In the above code snippet, we apply data free compression directly to the PyTorch model. 
However, you can also use [ct.optimize.coreml.prune_weights](https://apple.github.io/coremltools/source/coremltools.optimize.coreml.post_training_quantization.html#coremltools.optimize.coreml.prune_weights) if working with a Core ML model.

Trying with 50%, 75% and 90% `target_sparsity` we get `1.23ms`, `1.05ms` and `0.86ms` latency respectively.

Since we meet our latency goals with `90%` sparsity, we will now fine-tune the pretrained PyTorch model to see if we can get good enough accuracy.  

```python
from coremltools.optimize.torch.pruning import (
    MagnitudePruner,
    MagnitudePrunerConfig,
    ModuleMagnitudePrunerConfig
)
from coremltools.optimize.torch.pruning.pruning_scheduler import (
    PolynomialDecayScheduler
)

# Setup scheduler for applying sparsity during fine-tuning
scheduler = PolynomialDecayScheduler(update_steps=list(range(25000, 62500, 100)))
global_config = ModuleMagnitudePrunerConfig(target_sparsity=0.9, scheduler=scheduler)
config = MagnitudePrunerConfig().set_global(global_config)

pruner = MagnitudePruner(model, config)

pruner.prepare(inplace=True)

# Train the model
for epoch in range(num_epochs):
    model.train()
    for data, label in enumerate(train_loader):
        train_step(model, optimizer, train_loader, data, label, epoch)
        pruner.step()

model.eval()
pruned_model = pruner.finalize()
```

With fine-tuning, we observe an accuracy of `74.6%` for `90%` sparse ResNet50 model. This is quite good for this model, the results will vary based on the model.  

### Latency reduction with activation quantization 

In this section, we explore activation quantization on ResNet50 model, where we quantize both 
model weights and activations to 8-bit (W8A8). 
This can give latency gains by leveraging int8-int8 compute that is available on the Neural Engine (NE) on newer chips (A17 pro, M4).
           
You can apply activation quantization with calibration data using 
[`LinearQuantizer`](https://apple.github.io/coremltools/source/coremltools.optimize.torch.quantization.html#coremltools.optimize.torch.quantization.LinearQuantizer) 
API on the PyTorch model. 
We use the calibration data to measure statistics of activations and weights without actually 
simulating quantization during model's forward pass, and without needing to perform a backward pass.
Learn more about this workflow in the [API Overview](opt-quantization-api.md#calibration-data-based-quantization-1) 
section.

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

With 128 calibration samples, this gives us an accuracy of `76.1%` and a latency of `1.07ms`. 

Note that if you have a Core ML model, you can use the [`linear_quantize_activations`](opt-quantization-api.md#quantizing-weights-and-activations) 
method to quantize the activations.  

### Summary

Below we summarize results from above experiments as well as note the order of time taken to apply each of these compression workflows. 
With calibration data based activation quantization workflow we are able to achieve good speedup for minimal loss in accuracy, and it is much quicker to set up and apply.
On the other hand, by pruning the model with fine-tuning we are able to get even better speedup and a much smaller model for a slightly higher loss in accuracy, but with a more extensive set up. 

| Optimization API | Best config                | Accuracy | Model Size | Latency | Time to compress      |
|------------------|----------------------------|----------|------------|---------|-----------------------|
| Baseline         | -                          | 76.13%   | 48.8 MB    | 1.63ms  | -                     |
| MagnitudePruner  | 90% sparsity               | 74.60%   | 8.6 MB     | 0.86ms  | O(hours) (200 epochs) |
| LinearQuantizer  | W8A8 symmetric per-channel | 76.09%   | 25.8 MB    | 1.07ms  | O(minutes)            |

