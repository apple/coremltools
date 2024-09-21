Pruning
=======

Pruning a model is the process of sparsifying the weight matrices of the
model's layers, thereby reducing its storage size. You can also use pruning to reduce a
model's inference latency and power consumption. 

Magnitude Pruning
-----------------

.. autoclass:: coremltools.optimize.torch.pruning.ModuleMagnitudePrunerConfig
    :members: from_dict, as_dict, from_yaml

.. autoclass:: coremltools.optimize.torch.pruning.MagnitudePrunerConfig
    :members: set_global, set_module_type, set_module_name, from_dict, as_dict, from_yaml

.. autoclass:: coremltools.optimize.torch.pruning.MagnitudePruner
    :members: prepare, step, report, finalize


Pruning scheduler
-----------------

The :obj:`coremltools.optimize.torch.pruning.pruning_scheduler` submodule contains classes
that implement pruning schedules, which can be used for changing the
sparsity of pruning masks applied by various types of pruning algorithms
to prune neural network parameters.

.. autoclass:: coremltools.optimize.torch.pruning.pruning_scheduler.PruningScheduler
    :show-inheritance:
    :no-members:

.. autoclass:: coremltools.optimize.torch.pruning.pruning_scheduler.PolynomialDecayScheduler
    :show-inheritance:
    :members: compute_sparsity

.. autoclass:: coremltools.optimize.torch.pruning.pruning_scheduler.ConstantSparsityScheduler
    :show-inheritance:
    :members: compute_sparsity


SparseGPT
---------

.. autoclass:: coremltools.optimize.torch.layerwise_compression.LayerwiseCompressorConfig
    :members: from_dict, as_dict, from_yaml, get_layers

.. autoclass:: coremltools.optimize.torch.layerwise_compression.LayerwiseCompressor
    :members: compress

.. autoclass:: coremltools.optimize.torch.layerwise_compression.algorithms.ModuleSparseGPTConfig
    :show-inheritance:

.. autoclass:: coremltools.optimize.torch.layerwise_compression.algorithms.SparseGPT
    :show-inheritance:

