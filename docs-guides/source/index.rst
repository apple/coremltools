.. Core ML Tools Guide, created by
   sphinx-quickstart on Fri Jul 21 08:12:02 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Core ML Tools Guide
===================

.. image:: logo.png
   :alt: Core ML Tools logo
   :align: left
   :class: imgnoborder

Convert models from TensorFlow, PyTorch, and other libraries to Core ML. 
************************************************************************

The following guides include instructions and examples. For details about using the application programming interface classes and methods, see `API Reference <https://apple.github.io/coremltools/index.html>`_.

--------------

.. toctree::
   :maxdepth: 1
   :caption: Overview

   overview/overview-coremltools.md
   overview/new-features.md
   overview/faqs.md
   overview/coremltools-examples.md
   overview/migration.md
   overview/contributing.md


.. toctree::
   :maxdepth: 1
   :caption: Essentials

   essentials/installing-coremltools.md
   essentials/introductory-quickstart.md
   essentials/unified-conversion-api.md
   Core ML Model Format <https://apple.github.io/coremltools/mlmodel/index.html>
   API GitHub <https://github.com/apple/coremltools>


.. toctree::
   :maxdepth: 1
   :caption: Unified Conversion

   unified/convert-learning-models/convert-learning-models.rst
   unified/ml-programs/ml-programs.rst
   unified/convert-tensorflow/convert-tensorflow.rst
   unified/convert-pytorch/convert-pytorch.rst
   unified/conversion-options/conversion-options.rst
   unified/model-intermediate-language.md


.. toctree::
   :maxdepth: 1
   :caption: Optimization

   optimization/optimizing-models/optimizing-models.rst
   optimization/api-overview/api-overview.rst
   optimization/pruning/pruning.rst
   optimization/palettization/palettization.rst
   optimization/quantization-aware-training/quantization-aware-training.rst
   optimization/quantization-neural-network.md


.. toctree::
   :maxdepth: 1
   :caption: Other Converters

   other-converters/libsvm-conversion.md
   other-converters/sci-kit-learn-conversion
   other-converters/xgboost-conversion


.. toctree::
   :maxdepth: 1
   :caption: MLModel

   mlmodel/mlmodel.md
   mlmodel/xcode-model-preview-types.md
   mlmodel/mlmodel-utilities.md
   mlmodel/model-prediction.md
   mlmodel/updatable-model-examples/updatable-model-examples.rst


Index
-----

* :ref:`genindex`
* :ref:`search`
