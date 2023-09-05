Converting from PyTorch
=======================

You can convert a model trained in `PyTorch <https://pytorch.org>`_ to the Core ML format directly, without requiring an explicit step to save the PyTorch model in `ONNX format <https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html "Exporting a Model From PyTorch to ONNX">`_. Converting the model directly is recommended. (This feature was introduced in Core ML Tools version 4.0.)

.. toctree::
   :maxdepth: 1

   convert-pytorch-workflow.md
   model-tracing.md
   model-scripting.md
   convert-nlp-model.md
   convert-a-torchvision-model-from-pytorch.md
   pytorch-conversion-examples.md
