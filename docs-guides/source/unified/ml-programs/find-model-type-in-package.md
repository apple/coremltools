# Find the Model Type in a Model Package

If you need to determine whether an `mlpackage` file contains a neural network or an ML program, you can open it in Xcode 13, and look at the model type.

On a Linux system you can use Core ML Tools 5 or newer to inspect this property: 

```python
# load MLModel object
model = ct.models.MLModel("model.mlpackage")

# get the spec object
spec = model.get_spec()
print("model type: {}".format(spec.WhichOneof('Type')))
```

