Palettization
=============

Palettization is a mechanism for compressing a model by clustering the model's float
weights into a look-up table (LUT) of centroids and indices.

Palettization is implemented as an extension of `PyTorch's QAT <https://pytorch.org/docs/stable/quantization.html>`_
APIs. It works by inserting palettization layers in appropriate places inside a model.
The model can then be fine-tuned to learn the new palettized layers' weights in the form
of a LUT and indices. 

.. autoclass:: coremltools.optimize.torch.palettization.ModuleDKMPalettizerConfig
    :members: from_dict, as_dict, from_yaml

.. autoclass:: coremltools.optimize.torch.palettization.DKMPalettizerConfig
    :members: set_global, set_module_type, set_module_name, from_dict, as_dict, from_yaml

.. autoclass:: coremltools.optimize.torch.palettization.DKMPalettizer
    :members: prepare, step, report, finalize

.. autoclass:: coremltools.optimize.torch.palettization.ModulePostTrainingPalettizerConfig
.. autoclass:: coremltools.optimize.torch.palettization.PostTrainingPalettizer
.. autoclass:: coremltools.optimize.torch.palettization.PostTrainingPalettizerConfig
.. autoclass:: coremltools.optimize.torch.palettization.ModuleSKMPalettizerConfig
.. autoclass:: coremltools.optimize.torch.palettization.SKMPalettizer
.. autoclass:: coremltools.optimize.torch.palettization.SKMPalettizerConfig
