#  Copyright (c) 2024, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import itertools as _itertools
from typing import Callable as _Callable
from typing import List as _List
from typing import Optional as _Optional
from typing import Tuple as _Tuple

import torch as _torch
import torch.nn.functional as _F

_IS_TORCH_OLDER_THAN_2_3 = tuple(map(int, _torch.__version__.split(".")[:2])) < (2, 3)
_IS_TORCH_OLDER_THAN_2_4 = tuple(map(int, _torch.__version__.split(".")[:2])) < (2, 4)
if _IS_TORCH_OLDER_THAN_2_4:
    from torch.ao.quantization.pt2e.utils import get_aten_graph_module
else:
    from torch.ao.quantization.pt2e.utils import _get_aten_graph_module_for_pattern

from torch.ao.quantization.quantizer.quantizer import (
    FixedQParamsQuantizationSpec as _FixedQParamsQuantizationSpec,
)
from torch.ao.quantization.quantizer.quantizer import (
    QuantizationAnnotation as _QuantizationAnnotation,
)
from torch.ao.quantization.quantizer.quantizer import QuantizationSpec as _TorchQuantizationSpec
from torch.ao.quantization.quantizer.quantizer import (
    QuantizationSpecBase as _TorchQuantizationSpecBase,
)
from torch.ao.quantization.quantizer.quantizer import (
    SharedQuantizationSpec as _SharedQuantizationSpec,
)
from torch.ao.quantization.quantizer.xnnpack_quantizer import _get_module_name_filter
from torch.ao.quantization.quantizer.xnnpack_quantizer_utils import (
    _is_annotated,
    _mark_nodes_as_annotated,
)
from torch.fx import Node as _Node
from torch.fx.passes.utils.matcher_with_name_node_map_utils import (
    SubgraphMatcherWithNameNodeMap as _SubgraphMatcherWithNameNodeMap,
)
from torch.fx.passes.utils.source_matcher_utils import (
    get_source_partitions as _get_source_partitions,
)

from coremltools.optimize.torch.quantization._annotation_config import (
    AnnotationConfig as _AnnotationConfig,
)

# All activations recognized for conv-act/conv-bn-act patterns
_supported_activations = (
    _F.relu,
    _F.relu6,
    _F.leaky_relu,
    _F.silu,
    _F.elu,
    _F.celu,
    _F.selu,
    _F.mish,
    _F.hardtanh,
    _F.hardswish,
    _F.hardsigmoid,
)


# These activation functions don't have an inplace argument
_supported_activations_no_inplace = (_F.gelu, _F.sigmoid, _F.logsigmoid, _F.tanh)


# Map of dimension to convolution function
_conv_fn_map = {1: _F.conv1d, 2: _F.conv2d, 3: _F.conv3d}


def _get_aten_graph_module(
    pattern: _torch.nn.Module, example_inputs: _Tuple[_torch.Tensor], is_cuda: bool = False
):
    if _IS_TORCH_OLDER_THAN_2_3:
        return get_aten_graph_module(pattern.forward, example_inputs, is_cuda)
    elif _IS_TORCH_OLDER_THAN_2_4:
        return get_aten_graph_module(pattern, example_inputs, is_cuda)
    else:
        return _get_aten_graph_module_for_pattern(pattern, example_inputs, is_cuda)


def _adjust_activation_qspec(
    node: _torch.fx.Node, qspec: _Optional[_TorchQuantizationSpecBase]
) -> _Optional[_TorchQuantizationSpecBase]:
    """
    Adjust quantization spec for ops which can use fixed qparams
    or ops for which we can use affine quantization mode during
    symmetric quantization because their output is always positive.
    """
    if qspec is None:
        return qspec

    tanh_qspec = _FixedQParamsQuantizationSpec(
        dtype=_torch.uint8,
        scale=2.0 / 256.0,
        zero_point=128,
        quant_min=0,
        quant_max=255,
        qscheme=_torch.per_tensor_symmetric,
    )

    sigmoid_qspec = _FixedQParamsQuantizationSpec(
        dtype=_torch.uint8,
        scale=1.0 / 256.0,
        zero_point=0,
        quant_min=0,
        quant_max=255,
        qscheme=_torch.per_tensor_affine,
    )

    fixed_q_params_ops = {
        _torch.ops.aten.tanh.default: tanh_qspec,
        _torch.ops.aten.tanh_.default: tanh_qspec,
        _torch.ops.aten.sigmoid.default: sigmoid_qspec,
        _torch.ops.aten.sigmoid_.default: sigmoid_qspec,
        _torch.ops.aten.hardsigmoid.default: sigmoid_qspec,
        _torch.ops.aten.hardsigmoid_.default: sigmoid_qspec,
    }

    always_affine_ops = (
        _torch.ops.aten.relu.default,
        _torch.ops.aten.relu_.default,
        _torch.ops.aten.relu6.default,
        _torch.ops.aten.relu6_.default,
    )

    # ReLU6 activation maps to _torch.ops.aten.hardtanh.default with
    # min_val = 0 and max_val = 6
    is_always_affine_op = node.target in always_affine_ops or (
        node.target
        in [_torch.ops.aten.hardtanh.default, _torch.ops.aten.hardtanh_.default]
        and node.args[1] == 0  # min_val, corresponding to ReLU6
        and node.args[2] == 6  # max_val, corresponding to ReLU6
    )

    if node.target in fixed_q_params_ops:
        return _TorchQuantizationSpec(
            observer_or_fake_quant_ctr=qspec.observer_or_fake_quant_ctr,
            dtype=qspec.dtype,
            qscheme=fixed_q_params_ops[node.target].qscheme,
        )
        # FIXME: Because of a bug in PyTorch in function _create_obs_or_fq_from_qspec
        #        in module torch/ao/quantization/fx/prepare.py  which creates a
        #        FixedQParamsFakeQuantize partial, instead of an instance, we cannot
        #        actually create FixedQParamsQuantizationSpec
    if is_always_affine_op:
        return _TorchQuantizationSpec(
            observer_or_fake_quant_ctr=qspec.observer_or_fake_quant_ctr,
            dtype=qspec.dtype,
            qscheme=_torch.per_tensor_affine,
        )
    return qspec


def get_object_type_filter(tp: _Callable):
    """
    Returns a filter which returns True if a node in the final exported graph
    was created because of an object of type ``tp``
    """

    def object_type_filter(n: _Node) -> bool:
        # example: {
        #     'add_10': ('add', <function _operator.add(a, b, /)>)
        # }
        nn_module_stack = n.meta.get("nn_module_stack", {})
        types = [t for _, t in nn_module_stack.values()]
        source_fn_stack = n.meta.get("source_fn_stack", {})
        types.extend([t for _, t in source_fn_stack])
        return tp in types

    return object_type_filter


def get_not_object_type_or_name_filter(
    tp_list: _List[_Callable], module_name_list: _List[str]
) -> _Callable[[_Node], bool]:
    """
    Returns a filter which returns True if a node in the final exported graph
    was not created using any modules with names in ``module_name_list`` or objects with
    type in ``tp_list``.
    """
    object_type_filters = [get_object_type_filter(tp) for tp in tp_list]
    module_name_list_filters = [_get_module_name_filter(m) for m in module_name_list]

    def not_object_type_or_name_filter(n: _Node) -> bool:
        return not any(f(n) for f in object_type_filters + module_name_list_filters)

    return not_object_type_or_name_filter


def _get_weighted_mod_pattern(
    mod_fn: _Callable,
    example_inputs: _Tuple[_torch.Tensor, ...],
    act_fn: _Optional[_Callable] = None,
    act_in_place: bool = False,
) -> _torch.nn.Module:
    """
    Returns an aten graph corresponding to a sequence of these ops:
    input -> weighted_mod -> activation -> output

    A weighted mod is a module which has a weight and bias, such as a
    convolution module or a linear module.

    No activation is used if ``act_fn`` is ``None``.
    ``act_fn`` is an activation function from _supported_activations or
    _supported_activations_no_inplace
    """

    class Pattern(_torch.nn.Module):
        def forward(self, input, weight, bias):
            mod_out = mod_fn(input, weight, bias)
            output = mod_out
            node_dict = {
                "input": input,
                "mod": mod_out,
                "weight": weight,
                "bias": bias,
            }
            if act_fn is not None:
                # Only add output if activation function is applied to model output
                output = act_fn(output, inplace=True) if act_in_place else act_fn(output)
                node_dict["output"] = output
            return output, node_dict

    return _get_aten_graph_module(Pattern(), example_inputs)


def _get_weighted_mod_bn_pattern(
    mod_fn: _Callable,
    example_inputs: _Tuple[_torch.Tensor, ...],
    act_fn: _Optional[_Callable] = None,
    act_in_place: bool = False,
) -> _torch.nn.Module:
    """
    Returns an aten graph corresponding to a sequence of these ops:
    input -> weighted_mod -> batch_norm -> activation -> output

    A weighted mod is a module which has a weight and bias, such as a
    convolution module or a linear module.

    No activation is used if ``act_fn`` is ``None``.
    ``act_fn`` is an activation function from _supported_activations or
    _supported_activations_no_inplace
    """

    class Pattern(_torch.nn.Module):
        def forward(self, input, weight, bias, bn_weight, bn_bias, bn_run_mean, bn_run_var):
            mod_out = mod_fn(input, weight, bias)
            output = _F.batch_norm(
                mod_out, bn_run_mean, bn_run_var, bn_weight, bn_bias, training=True
            )
            if act_fn is not None:
                output = act_fn(output, inplace=True) if act_in_place else act_fn(output)
            return output, {
                "input": input,
                "mod": mod_out,
                "weight": weight,
                "bias": bias,
                "output": output,
            }

    return _get_aten_graph_module(Pattern(), example_inputs)


def get_binary_op_act_pattern(
    binary_op: _Callable,
    act_fn: _Optional[_Callable] = None,
    act_in_place: bool = False,
) -> _torch.nn.Module:
    """
    Returns an aten graph corresponding to a sequence of these ops:
    input_1 ---
               \
                --> binary_op -> activation -> output
               /
    input_2 ---

    A binary op is any operation which consumes two inputs to create one output,
    such as addition or multiplication.

    No activation is used if ``act_fn`` is ``None``.
    ``act_fn`` is an activation function from _supported_activations or
    _supported_activations_no_inplace
    """

    class Pattern(_torch.nn.Module):
        def forward(self, input_1, input_2):
            binary_op_out = binary_op(input_1, input_2)
            node_dict = {
                "binary_op": binary_op_out,
            }
            output = binary_op_out
            if act_fn is not None:
                output = act_fn(output, inplace=True) if act_in_place else act_fn(output)
                node_dict["output"] = output
            return output, node_dict

    example_inputs = (_torch.randn(1), _torch.randn(1))
    return _get_aten_graph_module(Pattern(), example_inputs)


def get_conv_pattern(
    conv_dim: int, act_fn: _Optional[_Callable] = None, act_in_place: bool = False
) -> _torch.nn.Module:
    """
    Returns an aten graph corresponding to a sequence of these ops:
    input -> conv -> activation -> output

    No activation is used if ``act_fn`` is ``None``.
    ``act_fn`` is an activation function from _supported_activations or
    _supported_activations_no_inplace
    """
    assert (
        conv_dim in _conv_fn_map
    ), f"Dimension {conv_dim} is not supported for Convolution layers."

    example_inputs = (
        _torch.randn(1, 1, *[3] * conv_dim),  # input
        _torch.randn(1, 1, *[1] * conv_dim),  # conv weight
        _torch.randn(1),  # conv bias
    )
    return _get_weighted_mod_pattern(
        _conv_fn_map[conv_dim], example_inputs, act_fn, act_in_place
    )


def get_conv_bn_pattern(
    conv_dim: int, act_fn: _Optional[_Callable] = None, act_in_place: bool = False
) -> _torch.nn.Module:
    """
    Returns an aten graph corresponding to a sequence of these ops:
    input -> conv -> batch_norm -> activation -> output

    No activation is used if ``act_fn`` is ``None``.
    ``act_fn`` is an activation function from _supported_activations or
    _supported_activations_no_inplace
    """
    assert (
        conv_dim in _conv_fn_map
    ), f"Dimension {conv_dim} is not supported for Convolution layers."

    example_inputs = (
        _torch.randn(1, 1, *[3] * conv_dim),  # input
        _torch.randn(1, 1, *[1] * conv_dim),  # conv weight
        _torch.randn(1),  # conv bias
        _torch.randn(1),  # bn_weight
        _torch.randn(1),  # bn_bias
        _torch.randn(1),  # bn_run_mean
        _torch.randn(1),  # bn_run_var
    )
    return _get_weighted_mod_bn_pattern(
        _conv_fn_map[conv_dim], example_inputs, act_fn, act_in_place
    )


def get_linear_pattern(
    act_fn: _Optional[_Callable] = None, act_in_place: bool = False
) -> _torch.nn.Module:
    """
    Returns an aten graph corresponding to a sequence of these ops:
    input -> linear -> activation -> output

    No activation is used if ``act_fn`` is ``None``.
    ``act_fn`` is an activation function from _supported_activations or
    _supported_activations_no_inplace
    """
    example_inputs = (
        _torch.randn(1, 1),  # input
        _torch.randn(3, 1),  # linear weight
        _torch.randn(3),  # linear bias
    )
    return _get_weighted_mod_pattern(_F.linear, example_inputs, act_fn, act_in_place)


def get_linear_bn_pattern(
    act_fn: _Optional[_Callable] = None, act_in_place: bool = False
) -> _torch.nn.Module:
    """
    Returns an aten graph corresponding to a sequence of these ops:
    input -> linear -> batch_norm -> activation -> output

    No activation is used if ``act_fn`` is ``None``.
    ``act_fn`` is an activation function from _supported_activations or
    _supported_activations_no_inplace
    """
    example_inputs = (
        _torch.randn(2, 1),  # input
        _torch.randn(3, 1),  # linear weight
        _torch.randn(3),  # linear bias
        _torch.randn(3),  # bn_weight
        _torch.randn(3),  # bn_bias
        _torch.randn(3),  # bn_run_mean
        _torch.randn(3),  # bn_run_var
    )
    return _get_weighted_mod_bn_pattern(_F.linear, example_inputs, act_fn, act_in_place)


def annotate_weighted_mod_pattern(
    model: _torch.fx.GraphModule,
    pattern_gm: _torch.fx.GraphModule,
    quantization_config: _Optional[_AnnotationConfig],
    filter_fn: _Optional[_Callable[[_Node], bool]],
) -> _Optional[_List[_List[_Node]]]:
    """
    Annotates all nodes in ``model``, which match the pattern specified by ``pattern_gm``
    using ``quantization_config``.

    ``pattern_gm`` captures patterns of the following type:

    input -> weighted_mod -> batch_norm -> activation -> output

    batch_norm and activation may or may not be applied in the pattern.

    Only annotates those patterns in which all nodes return True when ``filter_fn`` is applied
    to them.
    """
    model.graph.eliminate_dead_code()
    model.recompile()

    matcher = _SubgraphMatcherWithNameNodeMap(pattern_gm, ignore_literals=True)
    matches = matcher.match(model.graph)

    annotated_partitions = []
    for match in matches:
        name_node_map = match.name_node_map
        input_node = name_node_map["input"]
        mod_node = name_node_map["mod"]
        weight_node = name_node_map["weight"]
        bias_node = name_node_map["bias"]
        if "output" in name_node_map:
            # In this case, an activation is applied to the weighted module output
            output_node = name_node_map["output"]
            # If the output is same as mod_node, it means we have an inplace activation,
            # so we need to correct the mod_node
            if mod_node == output_node:
                mod_node = mod_node.args[0]
        else:
            output_node = None

        # Validate mod args
        if mod_node.args[0] is not input_node:
            raise ValueError(
                f"Weighted module arg did not contain input node {input_node}"
            )
        if mod_node.args[1] is not weight_node:
            raise ValueError(
                f"Weighted module arg did not contain weight node {weight_node}"
            )
        if len(mod_node.args) > 2 and mod_node.args[2] is not bias_node:
            raise ValueError(
                f"Weighted module arg did not contain bias node {bias_node}"
            )

        # Skip if the partition is already annotated or is filtered out by the user
        partition = [mod_node, weight_node]
        if bias_node is not None:
            partition.append(bias_node)
        if _is_annotated(partition):
            continue
        if filter_fn and any(not filter_fn(n) for n in partition):
            continue

        # Annotate conv inputs and pattern output
        input_qspec_map = dict()
        if not _is_annotated([input_node]):
            input_qspec_map[input_node] = (
                quantization_config.input_activation if quantization_config else None
            )
        else:
            input_qspec_map[input_node] = input_node.meta[
                "quantization_annotation"
            ].output_qspec

        input_qspec_map[weight_node] = (
            quantization_config.weight if quantization_config else None
        )
        output_qspec = (
            quantization_config.output_activation if quantization_config else None
        )
        if bias_node is not None:
            input_qspec_map[bias_node] = None

        if output_node is None:
            mod_node.meta["quantization_annotation"] = _QuantizationAnnotation(
                input_qspec_map=input_qspec_map,
                output_qspec=output_qspec,
                _annotated=True,
            )
        else:
            mod_node.meta["quantization_annotation"] = _QuantizationAnnotation(
                input_qspec_map=input_qspec_map,
                _annotated=True,
            )
            if not _is_annotated([output_node]):
                output_qspec = _adjust_activation_qspec(
                    node=output_node, qspec=output_qspec
                )
                output_node.meta["quantization_annotation"] = _QuantizationAnnotation(
                    output_qspec=output_qspec,
                    _annotated=True,
                )

        _mark_nodes_as_annotated(partition)
        annotated_partitions.append(partition)
    return annotated_partitions


def annotate_binary_op_act_pattern(
    model: _torch.fx.GraphModule,
    pattern_gm: _torch.fx.GraphModule,
    quantization_config: _Optional[_AnnotationConfig],
    filter_fn: _Optional[_Callable[[_Node], bool]] = None,
) -> _Optional[_List[_List[_Node]]]:
    """
    Annotates all nodes in ``model``, which match the pattern specified by ``pattern_gm``
    using ``quantization_config``.

    ``pattern_gm`` captures patterns of the following type:

    input_1 ---
               \
                --> binary_op -> activation -> output
               /
    input_2 ---

    activation may or may not be applied in the pattern.

    Only annotates those patterns in which all nodes return True when ``filter_fn`` is applied
    to them.
    """
    model.graph.eliminate_dead_code()
    model.recompile()

    matcher = _SubgraphMatcherWithNameNodeMap(pattern_gm, ignore_literals=True)
    matches = matcher.match(model.graph)

    annotated_partitions = []
    for match in matches:
        name_node_map = match.name_node_map
        binary_op_node: _Node = name_node_map["binary_op"]
        if "output" in name_node_map:
            output_node = name_node_map["output"]
            # In this case, an activation is applied to the weighted module output
            output_node = name_node_map["output"]
            # If the output is same as binary_op_node, it means we have an inplace activation,
            # so we need to correct the binary_op_node
            if binary_op_node == output_node:
                binary_op_node = binary_op_node.args[0]
            partition = [output_node, binary_op_node]
        else:
            output_node = None
            partition = [binary_op_node]

        if output_node is not None and len(binary_op_node.users) > 1:
            raise ValueError("Binary op with activation has more than one users.")

        if _is_annotated(partition):
            continue

        if filter_fn and any(not filter_fn(n) for n in partition):
            continue

        input_act_qspec = (
            quantization_config.input_activation if quantization_config else None
        )
        output_act_qspec = (
            quantization_config.output_activation if quantization_config else None
        )

        input_qspec_map = {}
        input_act0 = binary_op_node.args[0]
        if isinstance(input_act0, _Node):
            input_qspec_map[input_act0] = input_act_qspec

        input_act1 = binary_op_node.args[1]
        if isinstance(input_act1, _Node):
            input_qspec_map[input_act1] = input_act_qspec

        if output_node is None:
            binary_op_node.meta["quantization_annotation"] = _QuantizationAnnotation(
                input_qspec_map=input_qspec_map,
                output_qspec=output_act_qspec,
                _annotated=True,
            )
        else:
            binary_op_node.meta["quantization_annotation"] = _QuantizationAnnotation(
                input_qspec_map=input_qspec_map,
                _annotated=True,
            )
            output_act_qspec = _adjust_activation_qspec(
                node=output_node, qspec=output_act_qspec
            )
            output_node.meta["quantization_annotation"] = _QuantizationAnnotation(
                output_qspec=output_act_qspec,
                _annotated=True,
            )
        _mark_nodes_as_annotated(partition)
        annotated_partitions.append(partition)
    return annotated_partitions


def annotate_unary_shared_observer_ops(
    model: _torch.fx.GraphModule,
    ops: _List[_Callable],
    quantization_config: _Optional[_AnnotationConfig],
    filter_fn: _Optional[_Callable[[_Node], bool]] = None,
) -> _Optional[_List[_List[_Node]]]:
    """
    Annotates all nodes in ``model``, which correspond to unary ops specified in ``ops``.

    input --> op --> output

    input and output nodes share the same quantization parameters.
    """
    partitions = _get_source_partitions(model.graph, ops, filter_fn)
    annotated_partitions = []
    for _, op_partitions in partitions.items():
        for partition in op_partitions:
            output_node = partition.output_nodes[0]
            op_node = partition.nodes[0]
            if _is_annotated([output_node, op_node]):
                continue

            input_node = op_node.args[0]

            input_act_qspec = (
                quantization_config.input_activation if quantization_config else None
            )
            output_act_qspec = (
                quantization_config.output_activation if quantization_config else None
            )

            if (
                "quantization_annotation" not in input_node.meta
                or not input_node.meta["quantization_annotation"]._annotated
                or input_node.meta["quantization_annotation"].output_qspec is None
                or input_act_qspec is None
                or output_act_qspec is None
            ):
                continue

            # input and output of op will share quantization parameter with input of op
            act_qspec = _SharedQuantizationSpec(input_node)
            op_node.meta["quantization_annotation"] = _QuantizationAnnotation(
                input_qspec_map={
                    input_node: act_qspec,
                },
                _annotated=True,
            )
            output_node.meta["quantization_annotation"] = _QuantizationAnnotation(
                output_qspec=act_qspec,
                _annotated=True,
            )
            annotated_partitions.append(partition.nodes)
    return annotated_partitions


def annotate_conv_bn_act_helper(
    model: _torch.fx.GraphModule,
    quantization_config: _Optional[_AnnotationConfig],
    filter_fn: _Optional[_Callable[[_Node], bool]] = None,
    use_bn: bool = False,
) -> _Optional[_List[_List[_Node]]]:
    """
    A helper function for annotating all patterns involving convolution operations, i.e.,

    input -> conv -> batch_norm -> activation -> output

    conv can be either 1D, 2D or 3D
    batch_norm and activation may or may not be applied.
    """
    annotated_partitions = []

    pattern_map = {
        True: get_conv_bn_pattern,
        False: get_conv_pattern,
    }

    conv_dims = [1, 2, 3]
    combinations = _itertools.product(conv_dims, _supported_activations, [True, False])
    for conv_dim, act_fn, act_in_place in combinations:
        pattern_gm = pattern_map[use_bn](conv_dim, act_fn, act_in_place)
        annotated_partitions.extend(
            annotate_weighted_mod_pattern(
                model, pattern_gm, quantization_config, filter_fn
            )
        )

    combinations = _itertools.product(conv_dims, _supported_activations_no_inplace)
    for conv_dim, act_fn in combinations:
        pattern_gm = pattern_map[use_bn](conv_dim, act_fn, act_in_place=False)
        annotated_partitions.extend(
            annotate_weighted_mod_pattern(
                model, pattern_gm, quantization_config, filter_fn
            )
        )

    return annotated_partitions


def annotate_linear_bn_act_helper(
    model: _torch.fx.GraphModule,
    quantization_config: _Optional[_AnnotationConfig],
    filter_fn: _Optional[_Callable[[_Node], bool]] = None,
    use_bn: bool = False,
) -> _Optional[_List[_List[_Node]]]:
    """
    A helper function for annotating all patterns involving linear operations, i.e.,

    input -> linear -> batch_norm -> activation -> output

    batch_norm and activation may or may not be applied.
    """
    annotated_partitions = []

    pattern_map = {
        True: get_linear_bn_pattern,
        False: get_linear_pattern,
    }

    combinations = _itertools.product(_supported_activations, [True, False])
    for act_fn, act_in_place in combinations:
        pattern_gm = pattern_map[use_bn](act_fn, act_in_place)
        annotated_partitions.extend(
            annotate_weighted_mod_pattern(
                model, pattern_gm, quantization_config, filter_fn
            )
        )

    for act_fn in _supported_activations_no_inplace:
        pattern_gm = pattern_map[use_bn](act_fn, act_in_place=False)
        annotated_partitions.extend(
            annotate_weighted_mod_pattern(
                model, pattern_gm, quantization_config, filter_fn
            )
        )

    return annotated_partitions


def annotate_binary_op_helper(
    model: _torch.fx.GraphModule,
    binary_ops: _List[_Callable],
    quantization_config: _Optional[_AnnotationConfig],
    filter_fn: _Optional[_Callable[[_Node], bool]] = None,
) -> _Optional[_List[_List[_Node]]]:
    """
    A helper function for annotating all patterns involving binary operations, i.e.,
    using ``quantization_config``.

    input_1 ---
               \
                --> binary_op -> activation -> output
               /
    input_2 ---

    activation may or may not be applied in the pattern.
    """
    annotated_partitions = []

    combinations = _itertools.product(binary_ops, _supported_activations, [True, False])
    for binary_op, act_fn, act_in_place in combinations:
        pattern_gm = get_binary_op_act_pattern(binary_op, act_fn, act_in_place)
        annotated_partitions.extend(
            annotate_binary_op_act_pattern(
                model, pattern_gm, quantization_config, filter_fn
            )
        )

    combinations = _itertools.product(binary_ops, _supported_activations_no_inplace)
    for binary_op, act_fn in combinations:
        pattern_gm = get_binary_op_act_pattern(binary_op, act_fn, act_in_place=False)
        annotated_partitions.extend(
            annotate_binary_op_act_pattern(
                model, pattern_gm, quantization_config, filter_fn
            )
        )

    return annotated_partitions
