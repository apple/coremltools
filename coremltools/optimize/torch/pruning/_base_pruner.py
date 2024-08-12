#  Copyright (c) 2024, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import copy as _copy
import logging as _logging
from typing import Optional as _Optional
from typing import Tuple as _Tuple

import torch as _torch

from coremltools.optimize.torch._utils.metadata_utils import (
    register_metadata_version as _register_metadata_version,
)
from coremltools.optimize.torch._utils.torch_utils import get_eval_model as _get_eval_model
from coremltools.optimize.torch.base_model_optimizer import (
    BaseTrainingTimeModelOptimizer as _BaseTrainingTimeModelOptimizer,
)
from coremltools.optimize.torch.base_model_optimizer import _Report
from coremltools.optimize.torch.optimization_config import OptimizationConfig as _OptimizationConfig
from coremltools.optimize.torch.pruning import _utils
from coremltools.optimize.torch.pruning._base_pruning_method import BaseDynamicPruningMethod

_logger = _logging.getLogger(__name__)


class BasePruner(_BaseTrainingTimeModelOptimizer):
    pass


class BasePrunerWithPruningMethod(BasePruner):
    """
    Base class for all pruners which use a PruningMethod (implemented in

    """

    _supported_modules: _Tuple

    def __init__(self, model: _torch.nn.Module, config: _OptimizationConfig):
        super().__init__(model, config)
        self._pruner_info = {}

    @property
    def _is_prepared(self) -> bool:
        return len(self._pruner_info) > 0

    def prepare(self, inplace: bool = False) -> _torch.nn.Module:
        """
        Prepares the model for pruning.

        Args:
            inplace (:obj:`bool`): If ``True``, model transformations are carried out in-place and
                the original module is mutated, otherwise a copy of the model is mutated and returned.
        """
        return self._get_model_for_compression(inplace=inplace)

    def step(self):
        """
        Steps through the pruning schedule once. At every call to
        :meth:`.step`, an internal step counter is incremented by one.
        """
        raise NotImplementedError()

    def finalize(
        self, model: _Optional[_torch.nn.Module] = None, inplace: bool = False
    ) -> _torch.nn.Module:
        """
        Prepares the model for export. Removes pruning forward pre-hooks
        attached to submodules and commits pruning changes to pruned module parameters by
        multiplying the pruning masks with the parameter matrix.

        Args:
            model (:obj:`nn.Module`): model to finalize
            inplace (:obj:`bool`): If ``True``, model transformations are carried out in-place and
                the original module is mutated, otherwise a copy of the model is mutated and returned.
        """
        if model is None:
            model = self._model
        finalized_model = model if inplace else _copy.deepcopy(model)

        # Add compression metadata
        _register_metadata_version(finalized_model)
        for name, pruner_info in self._pruner_info.items():
            submodule = finalized_model.get_submodule(name)
            _utils.register_compression_metadata(submodule, pruner_info, self._supported_modules)

        # Remove pruning hooks
        for name, submodule in finalized_model.named_modules(remove_duplicate=True):
            if hasattr(submodule, "pruning_method"):
                submodule.pruning_method.remove(submodule)
            # If the module has been joint pruned + palettized, then palettizer finalize()
            # can remove pruning_method attribute but not the forward pre hook. So we explicitly remove it.
            elif name in self._pruner_info and _utils.is_palettized_module(
                self._pruner_info[name].module
            ):
                for k, hook in submodule._forward_pre_hooks.items():
                    if isinstance(hook, BaseDynamicPruningMethod):
                        del submodule._forward_pre_hooks[k]

        if model is None:
            self._model = finalized_model
        return finalized_model

    def report(self) -> _Report:
        """
        Returns a dictionary with important statistics related to current state of pruning.
        Each key in the dictionary corresponds to a module name and the value is a dictionary
        containing the statistics such as ``unstructured_weight_sparsity``,
        number of parameters, etc. Also contains a ``global`` key containing the same statistics
        aggregated over all the modules set up for pruning.
        """
        report = _Report()
        with _get_eval_model(self._model):
            with _torch.no_grad():
                # add submodule level sparsity summary
                total_num_params = 0
                for name, pruner_info in self._pruner_info.items():
                    submodule = pruner_info.module
                    if hasattr(submodule, "pruning_method"):
                        submod_config = pruner_info.config
                        num_params = getattr(submodule, submod_config.param_name).detach().numel()
                        summary = {"#params": int(num_params)}
                        summary.update(submodule.pruning_method.get_sparsity_summary(submodule))
                        total_num_params += num_params
                        report[name] = summary
                # get global sparsity summary
                global_summaries = {"#params": total_num_params}
                for sparsity_type in ["structured", "unstructured", "block2"]:
                    layer_numel = [val["#params"] for _, val in report.items()]
                    layer_sparsities = [
                        val[f"{sparsity_type}_weight_sparsity"] for _, val in report.items()
                    ]
                    global_summaries[
                        f"{sparsity_type}_weight_sparsity"
                    ] = _utils.get_global_sparsity_summaries(layer_sparsities, layer_numel)
                report["global"] = global_summaries
        return report


_allowed_granularity_values = ["per_scalar", "per_kernel", "per_channel", "per_layer"]
