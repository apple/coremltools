```{eval-rst}
.. index:: 
    single: PyTorch; model exporting
    single: model exporting
    single: ExportedProgram
```

# Model Exporting

The recommended way to generate ExportedProgram for your model is to use PyTorch's [torch.export.export](https://pytorch.org/docs/stable/export.html#torch.export.export). Exporting runs an example input tensor through your model, and captures the operations that are invoked as that input makes its way through the model's layers.

```{admonition} Exporting Limitations

Core ML Tools torch.export conversion support is newly added in 8.0, currently in beta state, in line with the export API status in PyTorch. As of Core ML Tools 8.0, representative models such as resnet, mobile net, vit, mobile bert, [openelm](convert-llm) can be converted, and total torch op translation test coverage is roughly ~60%. We would recommend to use this actively developing path

Also, torch.export has limitations, see [PyTorch elaboration](https://pytorch.org/docs/stable/export.html#limitations-of-torch-export)
```

## Requirements
This example requires [PyTorch](https://pytorch.org/) and Core ML Tools 8.0 or newer versions. Use the following commands:
```shell
pip install torch
pip install coremltools
```

At the time of creating this example, the author environment is
```text Output
torch              2.4.1
coremltools        8.0
```

## Export and Convert your Model
The following example builds a simple model from scratch and exports it to generate the ExportedProgram object needed by the converter. Follow these steps:

1. Define a simple layer module to reuse:

    ```python
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    # Define a simple layer module we will reuse in our network
    class Attention(nn.Module):
        def __init__(self):
            super().__init__()
            self.wq = nn.Linear(32, 64)
            self.wk = nn.Linear(32, 64)
            self.wv = nn.Linear(32, 64)
            self.wo = nn.Linear(64, 32)

        def forward(self, embedding):
            q = self.wq(embedding)
            k = self.wk(embedding)
            v = self.wv(embedding)
            attention = F.scaled_dot_product_attention(q, k, v)
            o = self.wo(attention)
            return o
    ```

2. Define a simple network consisting of several base layers:

    ```python
    # A simple network consisting of several base layers
    class Transformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = Attention()
            self.w1 = nn.Linear(32, 16)
            self.w2 = nn.Linear(16, 32)
            self.w3 = nn.Linear(32, 16)

        def forward(self, embedding):
            attention = self.attention(embedding)
            x = embedding + attention
            y = self.w2(F.silu(self.w1(x)) * self.w3(x))
            z = x + y
            return z
    ```

3. Instantiate the network:

    ```python
    # Instantiate the network
    model = Transformer()
    model.eval()
    ```

4. Define the example input, which is needed by exporter, and export the model.

    Run `torch.export.export` on your model with an example input, and save the resulting exported object. For an example input, you can use one sample of training or validation data, or even use randomly-generated data as shown in the following code snippet:

    ```python
    # Example input, needed by exporter
    example_input = (torch.rand(2, 32),)
    # Define dynamic shapes to be considered by exporter, if any
    batch_dim = torch.export.Dim(name="batch_dim", min=1, max=128)
    dynamic_shapes = {"embedding": {0: batch_dim}}
    # Generate ExportedProgram by exporting
    exported_model = torch.export.export(model, example_input, dynamic_shapes=dynamic_shapes)
    ```

5. Convert the exported model to Core ML:

    ```python
    import coremltools as ct

    mlmodel = ct.convert(exported_model)
    ```

## Difference from Tracing
For tracing, `ct.convert` requires the `inputs` arg from user. This is no longer the case for exporting, since the ExportedProgram object carries all name and shape and dtype info.

Subsequently, for exporting, dynamic shape is no longer specified through the `inputs` arg of `ct.convert`. Instead, user will need to [express dynamism in torch.export](https://pytorch.org/docs/stable/export.html#expressing-dynamism), which will then be automatically converted to Core ML [`RangeDim`](https://apple.github.io/coremltools/source/coremltools.converters.mil.input_types.html#rangedim). As of Core ML Tools 8.0, [`EnumeratedShapes`](https://apple.github.io/coremltools/source/coremltools.converters.mil.input_types.html#enumeratedshapes) is not supported through model exporting, yet.
