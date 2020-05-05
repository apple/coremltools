import torch
import torchvision
from coremltools.converters import convert
from coremltools.models import MLModel

"""
In this example, we'll instantiate a PyTorch classification model and convert
it to CoreML.
"""

"""
Here we instantiate our model. In a real use case this would be your trained
model.
"""
model = torchvision.models.mobilenet_v2()

"""
The next thing we need to do is generate TorchScript for the model. The easiest
way to do this is by tracing it. Note that if your model includes control flow,
you'll need to take a different approach and directly annotate the model
(documentation and examples of this process are in the works).
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
# Note that using the commented input would produce exactly the same result, it
# would just take longer.
# example_input = torch.rand(1, 3, 512, 512)

"""
Now we actually trace the model. This will produce the TorchScript that the
CoreML converter needs.
"""
traced_model = torch.jit.trace(model, example_input)

"""
Now with a TorchScript representation of the model, we can call the CoreML
converter. The converter also needs an example input to know what size inputs
to expect.

Note that the converter API is still evolving and *will* change in the future.
"""
mlmodel = convert(
    traced_model,
    inputs=[example_input],
    # debug=True,
    # If conversion fails with a message like 'Pytorch convert function for op x
    # not implemented', that means we haven't yet implemented all ops/layers your
    # model uses. Please file a radar so we can prioritize those ops. You can get
    # a complete list of all the supported and unsupported ops in your model by
    # running the converter with the debug=True flag.
)

"""
If conversion fails for a reason other than unimplemented ops, it's likely a
bug in the converter. Please file a radar with the model and the steps to repro
the conversion failure.
"""

"""
Now with a conversion complete, we can create an MLModel and run inference.
"""
mlmodel.save('/tmp/mobilenet_v2.mlmodel')
result = mlmodel.predict({"input.1": example_input.numpy()})
