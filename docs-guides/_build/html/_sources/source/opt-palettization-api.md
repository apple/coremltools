API Overview
=============

This page summarizes all the APIs that are available to palettize the weights of a model.
While there are several APIs available, different due to the input model format (Core ML or PyTorch model) and/or the
optimization workflow, they all follow a similar flow and have the following steps in common:

1. Defining a config object, which specifies the parameters of the algorithm. Most of these parameters are common 
to all APIs (e.g. number of bits to palettize, granularity, etc.).
  - The config can be defined either at a global level, to be applied to all the ops/modules
  in the model, or can be customized based on op-type or op-names.
  - The config object can be initialized either with a dictionary in code or in a yaml file that can be loaded from disk.
2. Invoking a method to compress, that takes in the config and the model.


## Palettizing a Core ML model
### Post-Training Palettization API example
Post-Training Palettization performs a K-Means operation on the supported weight matrices of a model that 
has already been converted to Core ML.

The following example shows `6-bit` palettization applied to all the ops that have more than 512 parameters.
This is controlled by setting the `weight_threshold` parameter to `512`.
```python
import coremltools as ct
import coremltools.optimize as cto

# load model
mlmodel = ct.models.MLModel(uncompressed_model_path)

# define op config 
op_config = cto.coreml.OpPalettizerConfig(nbits=6, weight_threshold=512)

# define optimization config by applying the op config globally to all ops 
config = cto.coreml.OptimizationConfig(global_config=op_config)

# palettize weights
compressed_mlmodel = cto.coreml.palettize_weights(mlmodel, config)
```
Some key parameters that the config accepts are:
- `n_bits` : This controls the number of clusters, which are `2^n_bits` .
- `weight_threshold`: Weight tensors that are smaller than this size are not palettized. Defaults to 2048.
- `mode`: Determine how the LUT is constructed by specifying either `kmeans`, `unique` or `uniform`. 
- `granularity` : Granularity for palettization. One of `per_tensor` or `per_grouped_channel`.
- `group_size`: The number of channels in a group.

There is also an option to customize the ops to palettize. More granular control can be achieved by using the 
`op_type_configs` and `op_name_configs` flags of OptimizationConfig. In order to get the names of the ops to customize, 
see the [get_weights_metadata()](mlmodel-utilities.md#get-weights-metadata)
utility, which provides detailed information about all the weights in the network, 
along with the ops each weight feeds into.

The following example shows `6-bit` palettization applied to all ops, with the exception that all the `linear` ops are set 
to `8-bits`, and two of the conv ops (named `conv1` and `conv3`) are omitted from palettization.

```python
import coremltools as ct
import coremltools.optimize as cto

mlmodel = ct.models.MLModel(uncompressed_model_path)

global_config = cto.coreml.OpPalettizerConfig(nbits=6)
linear_config = cto.coreml.OpPalettizerConfig(nbits=8)
config = cto.coreml.OptimizationConfig(
    global_config=global_config,
    op_type_configs={"linear": linear_config},
    op_name_configs={"conv1": None, "conv3": None},
)
compressed_mlmodel = cto.coreml.palettize_weights(mlmodel, config)
```

For more details, please follow the detailed API page for [coremltools.optimize.coreml.palettize_weights](https://apple.github.io/coremltools/source/coremltools.optimize.coreml.post_training_quantization.html#coremltools.optimize.coreml.palettize_weights).

## Palettizing a Torch model
### Post-Training Palettization API example
This is the same as the post-training palettization on a Core ML model, except this is done on a Torch model.
The following example shows `4-bit` palettization applied to all ops, with `granularity` set as `per_grouped_channel`. 
The `group_size` specified for this example would be `4` which means that each group of `4` channels would have one LUT.
```python
from model_utilities import get_torch_model

from coremltools.optimize.torch.palettization import PostTrainingPalettizer, \
                                                     PostTrainingPalettizerConfig

# load model
torch_model = get_torch_model()
palettization_config_dict = {
  "global_config": {"n_bits": 4, "granularity": "per_grouped_channel", "group_size": 4},
}
palettization_config = PostTrainingPalettizerConfig.from_dict(palettization_config_dict)
palettizer = PostTrainingPalettizer(torch_model, palettization_config)

palettized_torch_model = palettizer.compress()
```
Some key parameters that the config accepts are:
- `n_bits` : This controls the number of clusters, which are `2^n_bits` .
- `lut_dtype`: The dtype to use for representing each element in lookup tables. When value is `None`, no quantization is 
performed. Supported values are `torch.int8` and `torch.uint8`. Defaults to `None`.
- `granularity` : Granularity for palettization. One of `per_tensor` or `per_grouped_channel`.
- `group_size`: The number of channels in a group.
- `channel_axis`: The channel axis to form a group of channels. Only effective when granularity is `per_grouped_channel`.
- `cluster_dim`: The dimension of centroids for each lookup table.
- `enable_per_channel_scale`: When set to `True`, weights are normalized along the output channels using per-channel 
scales before being palettized.

For more details, please follow the detailed API page for [coremltools.optimize.torch.palettization.PostTrainingPalettizer](https://apple.github.io/coremltools/source/coremltools.optimize.torch.palettization.html#coremltools.optimize.torch.palettization.PostTrainingPalettizer)

### Sensitive K-Means Palettization API Example
This API implements the [Sensitive K-Means Algorithm](./opt-palettization-algos.md#sensitive-k-means). This algorithm 
requires calibration data as well as a loss function to compute parameter sensitivity.

The following example shows `4-bit` palettization applied to all ops, with the exception that all the `linear` ops are
set to `6-bits`, and two of the conv ops (named `conv1` and `conv3`) are omitted from palettization. 

The config in any of the algorithms described on this page can be created using a `yaml` file, too. The palettization
config would be described in the below `palettization_config.yaml` file:
```yaml
global_config:
  n_bits: 4
module_type_configs:
  Linear:
    n_bits: 6
module_name_configs:
  conv1: null
  conv3: null
calibration_nsamples: 64
```
The python script will now just load this `yaml` config and perform `SKM Palettization` as follows:
```python
from model_utilities import get_torch_model
from data_utils import get_calibration_data

from coremltools.optimize.torch.palettization import (SKMPalettizer, SKMPalettizerConfig)
import torch.nn.functional as F

torch_model = get_torch_model()

palettization_config = SKMPalettizerConfig.from_yaml('palettization_config.yaml')
# create the loss function
loss_fn = lambda mod, dat: F.nll_loss(mod(dat[0]), dat[1])
calibration_data = get_calibration_data()

palettizer = SKMPalettizer(torch_model, palettization_config)
palettized_torch_model = palettizer.compress(dataloader=calibration_data, loss_fn=loss_fn)
```
The parameters for the config in this algorithm are the same as 
[post-training palettization](#post-training-palettization-api-example). The `compress()` API, however, requires two 
additional parameters:
- `dataloader`: An iterable where each element is an input to the model to be compressed. Used for computing gradients 
of model weights. 
- `loss_fn`: A callable which takes the model and data as input and performs a forward pass on the model and computes 
the training loss.

For more details, please follow the detailed API page for [coremltools.optimize.torch.palettization.SKMPalettizer](https://apple.github.io/coremltools/source/coremltools.optimize.torch.palettization.html#coremltools.optimize.torch.palettization.SKMPalettizer).

### Differentiable K-Means Palettization API Example
Differentiable K-Means is a training-time palettization algorithm that performs attention based 
differentiable K-Means on the weight matrices. This plugs in directly into a user’s training pipeline and typically has 
higher data requirements.

To perform training-time palettization, these are the key steps:

1. Define a [DKMPalettizerConfig](https://apple.github.io/coremltools/source/coremltools.optimize.torch.palettization.html#coremltools.optimize.torch.palettization.DKMPalettizerConfig) config to specify the palettization parameters.
2. Initialize the palettizer object using [DKMPalettizer](https://apple.github.io/coremltools/source/coremltools.optimize.torch.palettization.html#coremltools.optimize.torch.palettization.DKMPalettizer).
3. Call the [prepare](https://apple.github.io/coremltools/source/coremltools.optimize.torch.palettization.html#coremltools.optimize.torch.palettization.DKMPalettizer.prepare) API to update the PyTorch model with palettization-friendly modules.
4. Run the usual training loop, with the addition of the [palettizer.step](https://apple.github.io/coremltools/source/coremltools.optimize.torch.palettization.html#coremltools.optimize.torch.palettization.DKMPalettizer.step) call.
5. Once the model has converged, use the [finalize](https://apple.github.io/coremltools/source/coremltools.optimize.torch.palettization.html#coremltools.optimize.torch.palettization.DKMPalettizer.finalize) API to prepare the model for conversion to Core ML.

The following example shows `2-bit` DKM palettization applied to all ops. However, here let’s assume our specific use 
case demands that we kick off palettization at the tenth training step. That can be achieved by specifying the 
`milestone` parameter as `10`.

```python
from model_utilities import get_torch_model
from training_utilities import train_step
from data_utils import get_dataloader

from coremltools.optimize.torch.palettization import (DKMPalettizer, DKMPalettizerConfig)
import torch.nn as nn

torch_model = get_torch_model()
dataloader = get_dataloader()
num_palettization_epochs = 2

palettization_config_dict = {
  "global_config": {"n_bits": 4, "milestone": 10},
}

palettization_config = DKMPalettizerConfig.from_dict(palettization_config_dict)
palettizer = DKMPalettizer(torch_model, palettization_config)

# Call the prepare API to insert palettization friendly modules into the model
palettizer.prepare(inplace=True)

torch_model.train()

for epoch in range(num_palettization_epochs):
  for batch_idx, (data, target) in enumerate(dataloader):
    train_step(data, target, torch_model)
    palettizer.step()

palettized_torch_model = palettizer.finalize()
```
Some key parameters that the config accepts are:
- `n_bits` : This controls the number of clusters, which are `2^n_bits` .
- `weight_threshold`: Weight tensors that are smaller than this size are not palettized. Defaults to `2048`.
- `granularity` : Granularity for palettization. One of `per_tensor` or `per_grouped_channel`.
- `group_size`: The number of channels in a group.
- `enable_per_channel_scale`: When set to `True`, weights are normalized along the output channels using per-channel.
- `milestone` : The number of times the 
[`palettizer.step`](https://apple.github.io/coremltools/source/coremltools.optimize.torch.palettization.html#coremltools.optimize.torch.palettization.DKMPalettizer.step) 
API has to be called before palettization is enabled. This number can be a training step number if the `palettizer.step` 
API is called once every training step, or it can be an epoch number if the `palettizer.step` API is called once every 
epoch. Defaults to `0`, in which case palettization is enabled from the start of the training loop.
- `cluster_dim`: The dimension of centroids for each lookup table.
- `quantize_activations`: When `True`, the activations are quantized.
- `quant_min`: The minimum value for each element in the weight clusters if they are quantized.
- `quant_max`: The maximum value for each element in the weight clusters if they are quantized.
- `dtype`: The dtype to use for quantizing the activations. Only applies when `quantize_activations` is `True`.
- `cluster_dtype`: The dtype to use for representing each element in lookup tables.

The DKM API has several other options, to control some of the specific knobs of the algorithm’s implementation. 
In most cases, you do not need to use values other than the default ones. To find out about these though, checkout the 
API Reference page [coremltools.optimize.torch.palettization.DKMPalettizer](https://apple.github.io/coremltools/source/coremltools.optimize.torch.palettization.html#coremltools.optimize.torch.palettization.DKMPalettizer).

[This notebook](https://apple.github.io/coremltools/_examples/dkm_palettization.html) provides a full example of applying DKM on an MNIST model.

## Converting the Palettized PyTorch Model
For a PyTorch model that has been palettized using `ct.optimize.torch.*` APIs, you can simply convert it using 
coremltools 8, without needing to specify any additional arguments. If you use any new feature,
such as `per_grouped_channel` `granularity` that is available in newer OS `iOS18/macOS15`, then you need to
specify the `minimum_deployment_target` flag accordingly:
```python
import torch
import coremltools as ct

palettized_torch_model.eval()
traced_model = torch.jit.trace(palettized_torch_model, example_input)

palettized_coreml_model = ct.convert(traced_model, inputs=...)

# or if iOS18 features were used in palettization 
palettized_coreml_model = ct.convert(traced_model, inputs=..,
                                     minimum_deployment_target=ct.target.iOS18)
```
If you palettized the Torch model using other non-coremltools APIs, or you are using coremltools version < 8, then 
please check out the [conversion page](./opt-conversion.md) to find out the process for getting the Core ML model with 
the correct palettized ops in it.
