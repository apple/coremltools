```{eval-rst}
.. index:: 
    single: neural network; compress weights
    single: quantization; neural network
```

# Compressing Neural Network Weights

```{admonition} For Neural Network Format Only

This page describes the API to compress the weights of a Core ML model that is of type `neuralnetwork`. For the [`mlprogram`](target-conversion-formats) model type, see the [Optimize API Overview](api-overview).
```

The Core ML Tools package includes a utility to compress the weights of a Core ML neural network model. Weight compression reduces the space occupied by the model. However, the precision of the intermediate tensors and the compute precision of the ops are not altered.

_Quantization_ refers to the process of reducing the number of bits that represent a number. The lower the number of bits, more the chances of degrading the model accuracy. The loss in accuracy varies with the model.

By default, the Core ML Tools converter produces a model with weights in floating-point 32 bit (float 32) precision. The weights can be quantized to 16 bits, 8 bits, 7 bits, and so on down to 1 bit. The intermediate tensors are kept in float precision (float 32 or float 16 depending on [execution unit](typed-execution)), while the weights are dequantized at runtime to match the precision of the intermediate tensors. Quantizing from float 32 to float 16 provides up to 2x savings in storage and generally does not affect the model's accuracy.

The [`quantize_weights`](https://apple.github.io/coremltools/source/coremltools.models.neural_network.html#coremltools.models.neural_network.quantization_utils.quantize_weights) function handles all quantization modes and options:

```python
from coremltools.models.neural_network import quantization_utils

# allowed values of nbits = 16, 8, 7, 6, ...., 1
quantized_model = quantization_utils.quantize_weights(model, nbits)
```

For a full list of supported arguments, see [`quantize_weights`](https://apple.github.io/coremltools/source/coremltools.models.neural_network.html#coremltools.models.neural_network.quantization_utils.quantize_weights) in the [API Reference for models.neural.network](https://apple.github.io/coremltools/source/coremltools.models.html#neural-network). The following examples demonstrate some of these arguments.

## Quantize to Float 16 Weights

Quantizing to float 16, which reduces by half the model's disk size, is the safest quantization option since it generally does not affect the model's accuracy:

```python
import coremltools as ct
from coremltools.models.neural_network import quantization_utils

# load full precision model
model_fp32 = ct.models.MLModel('model.mlmodel')

model_fp16 = quantization_utils.quantize_weights(model_fp32, nbits=16)
```

## Quantize to 1-8 Bits

Quantizing to 8 bits reduces the disk size to one fourth of the float 32 model. However, it may affect model accuracy, so you should always test the model after quantization, using test data. Depending on the model type, you may be able to quantize to bits lower than 8 without losing accuracy. 

```python
# quantize to 8 bit using linear mode
model_8bit = quantize_weights(model_fp32, nbits=8)

# quantize to 8 bit using LUT kmeans mode
model_8bit = quantize_weights(model_fp32, nbits=8,
                             quantization_mode="kmeans")

# quantize to 8 bit using linearsymmetric mode
model_8bit = quantize_weights(model_fp32, nbits=8,
                             quantization_mode="linear_symmetric")
```

When you set `nbits` to a value between 1 and 8, you can choose one of the following quantization modes:

- `linear`: The default mode, which uses linear quantization for weights with a scale and bias term.
- `linear_symmetric`: Symmetric quantization, with only a scale term.
- `kmeans_lut`: Uses a [k-means clustering](https://en.wikipedia.org/wiki/K-means_clustering) algorithm to construct a [lookup table](https://en.wikipedia.org/wiki/Lookup_table) (LUT) quantization of weights.

Try these different algorithms with your model, as some may work better than  others depending on the model type. 

## Quantization Options

The following options enable you to experiment with the quantization scheme so that you can find one that works best with your model. 

### Custom LUT Function

By default, the k-means algorithm is used to find the lookup table (LUT). However, you can provide a custom function to compute the LUT by setting `quantization_mode = "custom_lut "`. 

### Control Which Layers are Quantized

By default, all the layers that have weight parameters are quantized. However, the model accuracy may be sensitive to certain layers, which shouldn't be quantized. You can choose to skip quantization for certain layers and experiment as follows:

- Use the `AdvancedQuantizedLayerSelector` class, which lets you set simple properties such as layer types and weight count. For example:

```python
# Example: 8-bit symmetric linear quantization skipping bias,
# batchnorm, depthwise-convolution, and convolution layers
# with less than 4 channels or 4096 elements
from coremltools.models.neural_network.quantization_utils import AdvancedQuantizedLayerSelector

selector = AdvancedQuantizedLayerSelector(
    skip_layer_types=['batchnorm', 'bias', 'depthwiseConv'],
    minimum_conv_kernel_channels=4,
    minimum_conv_weight_count=4096
)

quantized_model = quantize_weights(model, 
                                   nbits=8,
                                   quantization_mode='linear_symmetric',
                                   selector=selector)
```

For a list of all the layer types in the Core ML neural network model, see the [`NeuralNetworkLayer`](https://apple.github.io/coremltools/mlmodel/Format/NeuralNetwork.html#neuralnetworklayer) section in the Core ML Format reference for [NeuralNetwork](https://apple.github.io/coremltools/mlmodel/Format/NeuralNetwork.html).

For finer control, you can write a custom rule to skip (or not skip) quantizing a layer by extending the `QuantizedLayerSelector` class:

```python
# Example : 8-bit linear quantization skipping the layer with name 'dense_2'
from coremltools.models.neural_network.quantization_utils import QuantizedLayerSelector


class MyLayerSelector(QuantizedLayerSelector):

    def __init__(self):
        super(MyLayerSelector, self).__init__()

    def do_quantize(self, layer, **kwargs):
        ret = super(MyLayerSelector, self).do_quantize(layer)
        if not ret or layer.name == 'dense_2':
            return True


selector = MyLayerSelector()
quantized_model = quantize_weights(
  mlmodel, 
  nbits = 8, 
  quantization_mode='linear', 
  selector=selector
)
```

