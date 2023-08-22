# Save ML Programs as Model Packages

The ML program type uses the [Core ML model package](save-as-model-package) container format that separates the model into components and offers more flexible metadata editing. Since an ML program decouples the weights from the program architecture, it cannot be saved as an `.mlmodel` file.

Use the `save()` method to save a file with the `.mlpackage` extension, as shown in the following example:

```python
model.save("my_model.mlpackage")
```

```{warning} Requires Xcode 13 and Newer

The model package format is supported on Xcode 13

```

