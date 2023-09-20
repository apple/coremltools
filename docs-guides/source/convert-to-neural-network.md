```{eval-rst}
.. index::
    pair: neural network; convert to
```


# Convert Models to Neural Networks

With versions of Core ML Tools older than 7.0, if you didn't specify the model type, or your `minimum_deployment_target` was a version older than iOS15, macOS12, watchOS8, or tvOS15, the model was converted by default to a neural network.

```{note}
To convert a model to the newer ML program model type, see [Convert Models to ML Programs](convert-to-ml-program).
```

To convert to a neural network using Core ML Tools version 7.0 or newer, specify the model type with the `convert_to` parameter, as shown in the following example:

```python
import coremltools as ct  # Core ML Tools version 7.0
# provide the "convert_to" argument to convert to a neural network
model = ct.convert(source_model, convert_to="neuralnetwork")
```

Alternatively, you can use the `minimum_deployment_target` parameter to specify a target such as `minimum_deployment_target=target.iOS14` or older, as shown in the following example: 

```python
import coremltools as ct  # Core ML Tools version 7.0
# provide the "minimum_deployment_target" argument to convert to a neural network
model = ct.convert(source_model, 
                   minimum_deployment_target=ct.target.iOS14)
```

