Optimizing Models
=================

With the increasing number of models deployed in a single app, and the size of the models also increasing, it is critical to optimize each model's memory footprint. To deploy models on devices such as the iPhone, you often need to optimize the models to use less storage space, reduce power consumption, and reduce latency during inference.

In this section you learn about model compression techniques that are compatible with the Core ML runtime and Apple hardware, and at what stages in your model deployment workflow can you compress your models. You also learn about the trade-offs, the APIs in Core ML Tools that can help you achieve these model optimizations, and the kind of memory savings and latency impact you can expect from different techniques.

.. toctree::
   :maxdepth: 1

   optimization-overview.md
   optimization-workflow.md
   performance-impact.md