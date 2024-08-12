#  Copyright (c) 2024, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from collections import defaultdict as _defaultdict
from typing import Any as _Any
from typing import Optional as _Optional
from typing import Tuple as _Tuple

import torch as _torch
import torch.ao.quantization as _aoquant
import torch.fx as _fx
import torch.nn as _nn
import torch.nn.intrinsic as _nni
import torch.nn.intrinsic.qat as _nniqat
from torch.ao.quantization.backend_config import BackendConfig as _BackendConfig
from torch.ao.quantization.fx.custom_config import PrepareCustomConfig as _PrepareCustomConfig
from torch.quantization.quantize_fx import prepare_qat_fx as _prepare_qat_fx

import coremltools.optimize.torch.quantization.modules.qat_modules as _qat
from coremltools.optimize.torch._utils.graph_utils import count_model_params as _count_model_params
from coremltools.optimize.torch._utils.torch_utils import (
    get_parent_child_name as _get_parent_child_name,
)
from coremltools.optimize.torch.quantization._backend_config import _fixed_qparams_modules
from coremltools.optimize.torch.quantization._utils import CombinationOpType as _CombinationOpType
from coremltools.optimize.torch.quantization._utils import combine_op_type as _combine_op_type
from coremltools.optimize.torch.quantization._utils import find_module as _find_module
from coremltools.optimize.torch.quantization._utils import find_target as _find_target
from coremltools.optimize.torch.quantization._utils import (
    get_share_qparams_ops as _get_share_qparams_ops,
)
from coremltools.optimize.torch.quantization._utils import (
    group_activation_quantization_modules_by_id as _group_activation_quantization_modules_by_id,
)
from coremltools.optimize.torch.quantization._utils import (
    is_activation_post_process as _is_activation_post_process,
)
from coremltools.optimize.torch.quantization._utils import is_quantized as _is_quantized
from coremltools.optimize.torch.quantization.modules.observers import NoopObserver as _NoopObserver
from coremltools.optimize.torch.quantization.quantization_config import (
    QuantizationScheme as _QuantizationScheme,
)
from coremltools.optimize.torch.quantization.quantization_config import (
    _default_quantization_options,
)

# layers which only scale the output and hence can use zero point = 0 if needed
_scale_only_layers = {
    _torch.nn.Dropout,
    _torch.nn.Dropout1d,
    _torch.nn.Dropout2d,
    _torch.nn.Dropout3d,
}


# layers which are always quantized with affine config because they have zero point = 0
_always_affine_layers = {
    _torch.nn.ReLU,
    _torch.nn.functional.relu,
    _torch.nn.functional.relu_,
    _torch.nn.ReLU6,
    _nni.ConvReLU1d,
    _nniqat.ConvReLU1d,
    _nni.ConvReLU2d,
    _nniqat.ConvReLU2d,
    _nni.ConvReLU3d,
    _nniqat.ConvBnReLU3d,
    _nni.ConvBnReLU1d,
    _nniqat.ConvBnReLU1d,
    _nni.ConvBnReLU2d,
    _nniqat.ConvBnReLU2d,
    _nni.ConvBnReLU3d,
    _nniqat.ConvBnReLU3d,
    _nni.LinearReLU,
    _nniqat.LinearReLU,
    _nni.BNReLU3d,
    _nni.BNReLU3d,
}

# fused quantized layers
_fused_quantized_layers = {
    _qat.ConvAct1d,
    _qat.ConvBnAct1d,
    _qat.ConvAct2d,
    _qat.ConvBnAct2d,
    _qat.ConvAct3d,
    _qat.ConvBnAct3d,
    _qat.ConvTransposeAct1d,
    _qat.ConvTransposeBnAct1d,
    _qat.ConvTransposeAct2d,
    _qat.ConvTransposeBnAct2d,
    _qat.ConvTransposeAct3d,
    _qat.ConvTransposeBnAct3d,
    _qat.LinearAct,
}

_common_observer_param_names = [
    "dtype",
    "qscheme",
    "reduce_range",
    "quant_min",
    "quant_max",
    "eps",
]

_observer_type_to_param_names = {
    _aoquant.MinMaxObserver: list(_common_observer_param_names),
    _aoquant.PerChannelMinMaxObserver: list(_common_observer_param_names) + ["ch_axis"],
    _aoquant.MovingAverageMinMaxObserver: list(_common_observer_param_names)
    + ["averaging_constant"],
    _aoquant.MovingAveragePerChannelMinMaxObserver: list(_common_observer_param_names)
    + ["averaging_constant", "ch_axis"],
    _aoquant.HistogramObserver: [
        "bins",
        "upsample_rate",
        "dtype",
        "qscheme",
        "reduce_range",
        "eps",
    ],
    _aoquant.PlaceholderObserver: [
        "dtype",
        "quant_min",
        "quant_max",
        "custom_op_name",
    ],
    _aoquant.NoopObserver: ["dtype", "custom_op_name"],
    _NoopObserver: ["dtype", "custom_op_name"],
    _aoquant.FixedQParamsObserver: [
        "scale",
        "zero_point",
        "dtype",
        "qscheme",
        "quant_min",
        "quant_max",
    ],
}


class QATConfigurationHandler:
    """
    Prepares the model for QAT by inserting weight and activation quantizers as
    specified in qconfig_mapping.

    Implements additional graph passes on a prepared module returned by prepare_qat_fx.
    """

    def __init__(
        self,
        prepare_custom_config: _PrepareCustomConfig,
        qconfig_mapping: _aoquant.QConfigMapping,
        backend_config: _BackendConfig,
        quantization_scheme: _QuantizationScheme,
    ):
        self._quantization_scheme = quantization_scheme
        self._qconfig_mapping = qconfig_mapping
        self._prepare_custom_config = prepare_custom_config
        self._backend_config = backend_config
        self._share_qparams_ops = _get_share_qparams_ops(self._backend_config)
        self._device = None
        self._act_quant_groups = dict()
        self._modules_to_replace = _defaultdict(list)
        self._new_act_post_process = dict()

    def prepare(self, model: _nn.Module, example_inputs: _Tuple[_Any, ...]):
        """
        Performs graph passes on model to configure activation and weight quantization layers.
        """
        model = _prepare_qat_fx(
            model,
            prepare_custom_config=self._prepare_custom_config,
            qconfig_mapping=self._qconfig_mapping,
            example_inputs=example_inputs,
            backend_config=self._backend_config,
        )

        self._setup_fake_quant_module_device(model, example_inputs)
        self._act_quant_groups = _group_activation_quantization_modules_by_id(model)
        if self._quantization_scheme == _QuantizationScheme.symmetric:
            self._mark_always_affine_layers_for_replacement(model)
            self._mark_always_affine_combination_ops_for_replacement(model)
        self._mark_fixed_qparams_modules_for_replacement(model)
        self._replace_weight_fake_quant_for_embedding_layers(model)
        model = self._replace_activation_quantizers(model)
        model = self._remove_activation_quantizer_after_dropout(model)
        return model

    def _setup_fake_quant_module_device(
        self, model: _fx.GraphModule, example_inputs: _Tuple[_Any, ...]
    ):
        """
        Set device for all fake quantize modules by inferring from model and/or data
        """
        # Record the device of the model
        count_params = _count_model_params(model)
        if count_params > 0:
            self._device = next(model.parameters()).device
        elif len(example_inputs) > 0:
            self._device = example_inputs[0].device
        else:
            self._device = _torch.device("cpu")

        for name, module in model.named_modules(remove_duplicate=True):
            if (
                hasattr(module, "weight_fake_quant")
                and module.weight_fake_quant is not None
                and hasattr(module, "set_device")
            ):
                module.weight_fake_quant.set_device(self._device)
            elif not name.endswith(".weight_fake_quant") and hasattr(module, "set_device"):
                module.set_device(self._device)

    def _get_affine_act_post_process_mod_from_symmetric(self, module: _aoquant.FakeQuantizeBase):
        """
        Returns activation post process module which is same as module but with
        affine qscheme instead of symmetric.
        """
        activation_post_process = module.activation_post_process
        observer_type = type(activation_post_process)
        if observer_type not in _observer_type_to_param_names:
            raise ValueError(f"Found unrecognized observer type {type(activation_post_process)}.")
        observer_param_names = _observer_type_to_param_names[observer_type]
        kwargs = {k: getattr(activation_post_process, k) for k in observer_param_names}
        if "qscheme" in kwargs:
            kwargs["qscheme"] = _torch.per_tensor_affine

        if module.ch_axis != -1:
            new_act_post_process = _aoquant.FakeQuantize(
                observer=observer_type, ch_axis=module.ch_axis, **kwargs
            )
        else:
            new_act_post_process = _aoquant.FakeQuantize(observer=observer_type, **kwargs)
        return new_act_post_process

    def _replace_activation_quantizers(self, model: _fx.GraphModule) -> _fx.GraphModule:
        """
        Replaces all nodes marked for replacement with new nodes.
        """
        replaced = set()
        for node, new_act_post_process in self._new_act_post_process.items():
            if node not in replaced:
                model.delete_submodule(node.target)
                model.add_submodule(node.target, new_act_post_process)
                replaced.add(node)
                # replace pointers to all modules which share this activation quantizer
                for child_node in self._modules_to_replace[node]:
                    if child_node not in replaced:
                        parent, child = _get_parent_child_name(child_node.target)
                        parent_mod = model.get_submodule(parent)
                        setattr(parent_mod, child, new_act_post_process)
                        replaced.add(child_node)
        model.recompile()
        return model

    def _mark_act_post_process_for_replacement(
        self,
        node: _fx.Node,
        model: _fx.GraphModule,
        new_act_post_process: _Optional[_aoquant.FakeQuantizeBase] = None,
    ):
        """
        Marks an activation post process layer (activation quantizer) for replacement.
        """
        shared_qparam_nodes = []
        if len(node.users) == 1:
            next_node = list(node.users.keys())[0]
            next_module = _find_module(model, next_node)
            if _is_activation_post_process(next_module) and _is_quantized(next_module):
                module_to_replace_id = id(model.get_submodule(next_node.target))
                # Some mods share the activation quantizer being replaced here,
                # so we collect all those mods here so that those can be pointed to
                # the new replaced module
                for child_node in self._act_quant_groups[module_to_replace_id]:
                    consumer_node = child_node.args[0]
                    if consumer_node.op == "call_module":
                        child_mod = _find_module(model, consumer_node)
                        if type(child_mod) in self._share_qparams_ops:
                            shared_qparam_nodes.append(child_node)
                            self._modules_to_replace[child_node] = []
                    elif consumer_node.op == "call_function":
                        if consumer_node.target in self._share_qparams_ops:
                            shared_qparam_nodes.append(child_node)
                            self._modules_to_replace[child_node] = []
                self._modules_to_replace[next_node] = shared_qparam_nodes
                if new_act_post_process is None:
                    new_act_post_process = self._get_affine_act_post_process_mod_from_symmetric(
                        next_module
                    )
                self._new_act_post_process[next_node] = new_act_post_process

    @staticmethod
    def _remove_activation_quantizer_after_dropout(model: _fx.GraphModule):
        """
        During evaluation, dropout is a no-op. During conversion,

        conv_1 -> activation_q_1 -> dropout -> activation_q_2 -> conv_2

        becomes

        conv_1 -> quant_1 -> dequant_1 -> quant_2 -> dequant_2 -> conv_2

        where quant_1,dequant_1 have different qparams from quant_2/dequant_2
        because dropout scales the output by 1/(1-p). This leads to inefficiency
        during inference. Since during inference, conv_2 sees quantized activations
        coming from conv_1, removing activation_q_2 doesn't lead to
        increased quantization error. Hence, this pass removes activation_q_2.
        """
        nodes_to_remove = set()
        for node in model.graph.nodes:
            if node.op == "call_module":
                layer = _find_module(model, node)
                if isinstance(layer, tuple(_scale_only_layers)):
                    prev_module = _find_module(model, node.prev)
                    next_module = _find_module(model, node.next)
                    if _is_activation_post_process(next_module) and _is_activation_post_process(
                        prev_module
                    ):
                        nodes_to_remove.add(node.next)
        for node in nodes_to_remove:
            node.replace_all_uses_with(node.prev)
            model.delete_submodule(node.target)
            model.graph.erase_node(node)
        model.recompile()
        return model

    def _mark_always_affine_layers_for_replacement(self, model: _fx.GraphModule):
        """
        Some layers like ReLU can be quantized with affine qscheme even when we want
        to use symmetric quantization (zero point = 0). This is because these layers
        always have a non-negative output. And thus, an affine activation post process layer attached
        after layers like these will always observe zero point as 0. This can possibly help us
        reduce quantization error because of the larger number of quantization levels available.
        (Symmetric quantization will force the output of these layers to use [0, 127] as the
        output range, but with affine quantization, we can use [0, 255]).

        prepare_qat_fx requires all modules being fused together to have the same QConfig.
        Thus, if we have a Conv followed by a ReLU and we want to set ReLU to have affine qscheme,
        we would have to set Conv to use affine qscheme as well. But this isn't correct because a stand alone
        Conv layer somewhere else in the network will also use affine qscheme which is undesirable
        we want to fix zero point to 0.

        Hence, we add this pass which replaces all occurrences of activation post process after
         ``always_affine_layers`` with an affine version.
        """
        # Note: For all these ops, whether or not we can use affine qscheme for them depends only on
        # the op itself or one preceding op.
        # Note: graph.nodes traverses the nodes in topological order
        for node in model.graph.nodes:
            if node.op == "call_module":
                layer = _find_target(model, node.target)
                if type(layer) in _always_affine_layers:
                    self._mark_act_post_process_for_replacement(node, model)
                elif isinstance(layer, tuple(_fused_quantized_layers)):
                    if type(layer.act) in _always_affine_layers:
                        self._mark_act_post_process_for_replacement(node, model)
                # layers which only scale the output can also use affine qcheme
                elif isinstance(layer, tuple(_scale_only_layers)):
                    arg_mod = _find_module(model, node.args[0])
                    if (
                        _is_activation_post_process(arg_mod)
                        and node.args[0] in self._modules_to_replace
                    ):
                        self._mark_act_post_process_for_replacement(node, model)
            elif node.op == "call_function":
                combine_op_type = _combine_op_type(node)
                if combine_op_type is not None:
                    if combine_op_type == _CombinationOpType.AddReLU:
                        self._mark_act_post_process_for_replacement(node, model)
                elif node.target in _always_affine_layers:
                    self._mark_act_post_process_for_replacement(node, model)

    def _mark_always_affine_combination_ops_for_replacement(self, model: _fx.GraphModule):
        """
        This method follows the same reasoning as described in ``_mark_always_affine_layers_for_replacement``,
        but instead of replacing activation quantizers for stand-alone ops, it replaces them for
        ops which consume more than 1 tensor as input.

        For add or cat, if the qscheme of all tensors being combined together is
        affine, it also uses affine qscheme, otherwise, it uses symmetric qscheme.
        """
        for node in model.graph.nodes:
            if node.op == "call_function":
                combine_op_type = _combine_op_type(node)
                if combine_op_type is not None and combine_op_type != _CombinationOpType.AddReLU:
                    args = node.args
                    if combine_op_type == _CombinationOpType.Concat:
                        args = node.args[0]
                    arg_act_qschemes = []
                    for arg in args:
                        arg_mod = _find_module(model, arg)
                        if arg_mod is not None:
                            if (
                                type(arg_mod) in _always_affine_layers
                                or arg in self._modules_to_replace
                            ):
                                arg_act_qschemes.append(_QuantizationScheme.affine)
                            elif hasattr(arg_mod, "qscheme"):
                                if arg_mod.qscheme == _torch.per_tensor_affine:
                                    arg_act_qschemes.append(_QuantizationScheme.affine)
                                else:
                                    arg_act_qschemes.append(_QuantizationScheme.symmetric)
                            else:
                                arg_act_qschemes.append(_QuantizationScheme.symmetric)
                        else:
                            arg_act_qschemes.append(_QuantizationScheme.symmetric)
                    if all(x == _QuantizationScheme.affine for x in arg_act_qschemes):
                        # We have already marked cat op for replacement, when one of the
                        # tensors it combines was marked for replacement. So we don't need to
                        # add it here again.
                        if combine_op_type != _CombinationOpType.Concat:
                            self._mark_act_post_process_for_replacement(node, model)
                    else:
                        # If any of the tensor being cat-ed together need to use
                        # [-128, 127] range, we can't use affine quantization in
                        # symmetric mode for them, so we remove them from modules marked for replacement.
                        if combine_op_type == _CombinationOpType.Concat:
                            for arg in args:
                                if arg in self._modules_to_replace:
                                    self._modules_to_replace.pop(arg)
                                if arg in self._new_act_post_process:
                                    self._new_act_post_process.pop(arg)

    def _mark_fixed_qparams_modules_for_replacement(self, model: _fx.GraphModule):
        """
        If a fixed qparams activation is fused, with conv/linear, we need to make sure
        its qconfig is inherited by the fused op's activation quantizer. Before this step,
        all fused layers will have symmetric/affine activation quantizer.
        """
        for node in model.graph.nodes:
            if node.op == "call_module":
                layer = _find_target(model, node.target)
                if isinstance(layer, tuple(_fused_quantized_layers)):
                    # If output of this layer is being cat with another layer, we don't want
                    # to enforce that layer to use the same activation quantizer, so we ignore it
                    if _torch.cat in [
                        child_node.target for child_node in self._act_quant_groups[id(layer)]
                    ]:
                        continue
                    elif type(layer.act) in _fixed_qparams_modules:
                        act_post_process = self._qconfig_mapping.object_type_qconfigs[
                            type(layer.act)
                        ].activation()
                        self._mark_act_post_process_for_replacement(node, model, act_post_process)

    def _replace_weight_fake_quant_for_embedding_layers(self, model: _fx.GraphModule):
        """
        Changes qscheme of embedding layers from float qparams to integer qparams.
        """
        for node in model.graph.nodes:
            if node.op == "call_module":
                layer = _find_target(model, node.target)
                if (
                    isinstance(layer, _torch.nn.Embedding)
                    and hasattr(layer, "weight_fake_quant")
                    and isinstance(layer.weight_fake_quant, _aoquant.FakeQuantize)
                ):
                    weight_dtype = layer.weight_fake_quant.dtype
                    delattr(layer, "weight_fake_quant")

                    observer_cls = type(layer.qconfig.weight().activation_post_process)

                    layer.weight_fake_quant = _aoquant.FakeQuantize(
                        observer=observer_cls,
                        dtype=weight_dtype,
                        qscheme=_QuantizationScheme.get_qscheme(
                            self._quantization_scheme, is_per_channel=True
                        ),
                        ch_axis=_default_quantization_options["weight_ch_axis"],
                    )
