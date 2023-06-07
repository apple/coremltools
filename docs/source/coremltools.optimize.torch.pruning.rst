Training-Time Pruning
=====================

Pruning a model is the process of sparsifying the weight matrices of the
model's layers, thereby reducing its storage size. You can also use pruning to reduce a
model's inference latency and power consumption. 

Magnitude Pruning
-----------------

.. automodule:: coremltools.optimize.torch.pruning

    .. autoclass:: ModuleMagnitudePrunerConfig

    .. autoclass:: MagnitudePrunerConfig

    .. autoclass:: MagnitudePruner


Pruning scheduler
-----------------

The :obj:`coremltools.optimize.torch.pruning.pruning_scheduler` submodule contains classes
that implement pruning schedules, which can be used for changing the
sparsity of pruning masks applied by various types of pruning algorithms
to prune neural network parameters.

.. automodule:: coremltools.optimize.torch.pruning.pruning_scheduler

    .. autoclass:: PruningScheduler
        :show-inheritance:
        :no-members:

    .. autoclass:: PolynomialDecayScheduler
        :show-inheritance:
        :members: compute_sparsity

    .. autoclass:: ConstantSparsityScheduler
        :show-inheritance:
        :members: compute_sparsity
