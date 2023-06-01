Optimizers
===============================================

To deploy models on devices such as the iPhone, you often need to optimize the models to
use less storage space, reduce power consumption, and reduce latency during inference.
For an overview, see Optimizing Models Post-Training
(`Compressing ML Program Weights <https://coremltools.readme.io/docs/compressing-ml-program-weights>`_
and `Compressing Neural Network Weights <https://coremltools.readme.io/docs/quantization>`_).

coreml
---------------------------------------------------------

Optimizers that compress Core ML models:

.. toctree::
   :maxdepth: 1
   
   coremltools.optimize.coreml.post_training_quantization.rst
   coremltools.optimize.coreml.config.rst
   coremltools.optimize.coreml.graph.rst
   

