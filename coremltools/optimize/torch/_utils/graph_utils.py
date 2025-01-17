#  Copyright (c) 2024, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import operator as _operator
from typing import Any as _Any
from typing import Dict as _Dict
from typing import Iterable as _Iterable
from typing import List as _List
from typing import Tuple as _Tuple

import torch as _torch

from coremltools.optimize.torch._utils.registry import BaseRegistry as _BaseRegistry
from coremltools.optimize.torch._utils.transforms import _LINEAR_CONV_DECONV_FUNCS
from coremltools.optimize.torch._utils.transforms import ReplacementType as _ReplacementType
from coremltools.optimize.torch._utils.transforms import TransformRegistry as _TransformRegistry
from coremltools.optimize.torch._utils.transforms import fetch_argument as _fetch_argument
from coremltools.optimize.torch._utils.transforms import fetch_attr as _fetch_attr
from coremltools.optimize.torch._utils.transforms import fetch_func_params as _fetch_func_params
from coremltools.optimize.torch._utils.transforms import load_arg as _load_arg

from coremltools._deps import _HAS_TORCH_VISION, MSG_TORCH_VISION_NOT_FOUND

if _HAS_TORCH_VISION:
    import torchvision as _torchvision


def count_model_params(model: _torch.nn.Module) -> int:
    """
    Return the number of trainable parameters in the provided model.
    """
    return sum(param.numel() for param in model.parameters())


def generate_env(
    model: _torch.nn.Module,
    traced_model: _torch.fx.GraphModule,
    input_shape: _Tuple[int],
) -> _Dict[str, _Any]:
    """
    Computes environment dictionary by going through the provided model and
    traced_model with specified input_shape.
    The environment dictionary maps from node name to the actual
    function/method/module/attr/placeholder executed by the node.
    Returns environment dictionary.
    """

    env = dict()
    sample_input = _torch.rand(input_shape)
    count_params = count_model_params(model)
    if count_params > 0 and next(model.parameters()).is_cuda:
        sample_input = sample_input.cuda()
    modules = dict(model.named_modules())

    for node in traced_model.graph.nodes:
        if node.op == "placeholder":
            # Only one graph-level input
            result = sample_input
        elif node.op == "get_attr":
            result = _fetch_attr(model, node.target)
        elif node.op == "call_function":
            result = node.target(
                *_load_arg(node.args, env), **_load_arg(node.kwargs, env)
            )
        elif node.op == "call_method":
            self_obj, *args = _load_arg(node.args, env)
            kwargs = _load_arg(node.kwargs, env)
            result = getattr(self_obj, node.target)(*args, **kwargs)
        elif node.op == "call_module":
            ld_args = _load_arg(node.args, env)
            result = modules[node.target](
                *_load_arg(node.args, env), **_load_arg(node.kwargs, env)
            )
        env[node.name] = result
    return env


def volume(items: _Iterable[int]) -> int:
    """
    Given an input shape, computes the volume or total number of elements contained
    by that shape. Note that an empty tensor will always have a volume of 0.
    """

    ret = 1 if len(items) >= 1 else 0
    for item in items:
        ret *= item
    return ret


def num_bytes(dtype: _torch.dtype) -> int:
    """
    Computes the number of bytes required to represent the input datatype.
    """

    num_bits_in_byte = 8
    if dtype.is_floating_point:
        return _torch.finfo(dtype).bits / num_bits_in_byte
    else:
        return _torch.iinfo(dtype).bits / num_bits_in_byte


def parse_call_function_target(target: _Any) -> str:
    """
    Parses the call function target for the function-name and returns it.
    Initially, will try to lookup the function inside a dictionary that maps
    known functions to function-names. If the function is not found in the dictionary,
    then will use custom parsing code to get the function-name.
    """

    if not _HAS_TORCH_VISION:
        raise ImportError(MSG_TORCH_VISION_NOT_FOUND)

    func_map = {
        _torch.nn.functional.conv1d: "conv1d",
        _torch.nn.functional.conv2d: "conv2d",
        _torch.nn.functional.conv3d: "conv3d",
        _torch.nn.functional.conv_transpose1d: "conv_transpose1d",
        _torch.nn.functional.conv_transpose2d: "conv_transpose2d",
        _torch.nn.functional.conv_transpose3d: "conv_transpose3d",
        _torch.nn.functional.unfold: "unfold",
        _torch.nn.functional.fold: "fold",
        _torch.nn.functional.avg_pool1d: "avg_pool1d",
        _torch.nn.functional.avg_pool2d: "avg_pool2d",
        _torch.nn.functional.avg_pool3d: "avg_pool3d",
        _torch.nn.functional.max_pool1d: "max_pool1d",
        _torch.nn.functional.max_pool2d: "max_pool2d",
        _torch.nn.functional.max_pool3d: "max_pool3d",
        _torch.nn.functional.max_unpool1d: "max_unpool1d",
        _torch.nn.functional.max_unpool2d: "max_unpool2d",
        _torch.nn.functional.max_unpool3d: "max_unpool3d",
        _torch.nn.functional.lp_pool1d: "lp_pool1d",
        _torch.nn.functional.lp_pool2d: "lp_pool2d",
        _torch.nn.functional.adaptive_max_pool1d: "adaptive_max_pool1d",
        _torch.nn.functional.adaptive_max_pool2d: "adaptive_max_pool2d",
        _torch.nn.functional.adaptive_max_pool3d: "adaptive_max_pool3d",
        _torch.nn.functional.adaptive_avg_pool1d: "adaptive_avg_pool1d",
        _torch.nn.functional.adaptive_avg_pool2d: "adaptive_avg_pool2d",
        _torch.nn.functional.adaptive_avg_pool3d: "adaptive_avg_pool3d",
        _torch.nn.functional.fractional_max_pool2d: "fractional_max_pool2d",
        _torch.nn.functional.fractional_max_pool3d: "fractional_max_pool3d",
        _torch.nn.functional.scaled_dot_product_attention: "scaled_dot_product_attention",
        _torch.nn.functional.threshold: "threshold",
        _torch.nn.functional.threshold_: "threshold_",
        _torch.nn.functional.relu: "relu",
        _torch.nn.functional.relu_: "relu_",
        _torch.nn.functional.hardtanh: "hardtanh",
        _torch.nn.functional.hardtanh_: "hardtanh_",
        _torch.nn.functional.hardswish: "hardswish",
        _torch.nn.functional.relu6: "relu6",
        _torch.nn.functional.elu: "elu",
        _torch.nn.functional.elu_: "elu_",
        _torch.nn.functional.selu: "selu",
        _torch.nn.functional.celu: "celu",
        _torch.nn.functional.leaky_relu: "leaky_relu",
        _torch.nn.functional.leaky_relu_: "leaky_relu_",
        _torch.nn.functional.prelu: "prelu",
        _torch.nn.functional.rrelu: "rrelu",
        _torch.nn.functional.rrelu_: "rrelu_",
        _torch.nn.functional.glu: "glu",
        _torch.nn.functional.gelu: "gelu",
        _torch.nn.functional.logsigmoid: "logsigmoid",
        _torch.nn.functional.hardshrink: "hardshrink",
        _torch.nn.functional.tanhshrink: "tanhshrink",
        _torch.nn.functional.softsign: "softsign",
        _torch.nn.functional.softplus: "softplus",
        _torch.nn.functional.softmin: "softmin",
        _torch.nn.functional.softmax: "softmax",
        _torch.nn.functional.softshrink: "softshrink",
        _torch.nn.functional.gumbel_softmax: "gumbel_softmax",
        _torch.nn.functional.log_softmax: "log_softmax",
        _torch.nn.functional.tanh: "tanh",
        _torch.nn.functional.sigmoid: "sigmoid",
        _torch.nn.functional.hardsigmoid: "hardsigmoid",
        _torch.nn.functional.silu: "silu",
        _torch.nn.functional.mish: "mish",
        _torch.nn.functional.batch_norm: "batch_norm",
        _torch.nn.functional.group_norm: "group_norm",
        _torch.nn.functional.instance_norm: "instance_norm",
        _torch.nn.functional.layer_norm: "layer_norm",
        _torch.nn.functional.local_response_norm: "local_response_norm",
        _torch.nn.functional.normalize: "normalize",
        _torch.nn.functional.linear: "linear",
        _torch.nn.functional.bilinear: "bilinear",
        _torch.nn.functional.dropout: "dropout",
        _torch.nn.functional.alpha_dropout: "alpha_dropout",
        _torch.nn.functional.feature_alpha_dropout: "feature_alpha_dropout",
        _torch.nn.functional.dropout1d: "dropout1d",
        _torch.nn.functional.dropout2d: "dropout2d",
        _torch.nn.functional.dropout3d: "dropout3d",
        _torch.nn.functional.embedding: "embedding",
        _torch.nn.functional.embedding_bag: "embedding_bag",
        _torch.nn.functional.one_hot: "one_hot",
        _torch.nn.functional.pairwise_distance: "pairwise_distance",
        _torch.nn.functional.cosine_similarity: "cosine_similarity",
        _torch.nn.functional.pdist: "pdist",
        _torch.nn.functional.pixel_shuffle: "pixel_shuffle",
        _torch.nn.functional.pixel_unshuffle: "pixel_unshuffle",
        _torch.nn.functional.pad: "pad",
        _torch.nn.functional.interpolate: "interpolate",
        _torch.nn.functional.upsample: "upsample",
        _torch.nn.functional.upsample_nearest: "upsample_nearest",
        _torch.nn.functional.upsample_bilinear: "upsample_bilinear",
        _torch.nn.functional.grid_sample: "grid_sample",
        _torch.nn.functional.affine_grid: "affine_grid",
        _torch.Size: "Size",
        _torch.cat: "cat",
        _torch.concat: "concat",
        _torch.flatten: "flatten",
        _torch.add: "add",
        _torch.mul: "mul",
        _torch.sub: "sub",
        _torch.div: "div",
        _torchvision.ops.stochastic_depth: "stochastic_depth",
        _torch.permute: "permute",
        _torch.swapaxes: "swapaxes",
        _torch.einsum: "einsum",
        _torch.squeeze: "squeeze",
        _torch.unsqueeze: "unsqueeze",
        _torch.floor_divide: "floor_divide",
        _torch.Tensor.size: "size",
        _torch.Tensor.view: "view",
        _torch.Tensor.contiguous: "contiguous",
        _torch.transpose: "transpose",
        _torch.chunk: "chunk",
        _torch.mean: "mean",
        _torch.eq: "eq",
        _torch.Tensor.eq: "eq",
        _torch._assert: "_assert",
        _torch.reshape: "reshape",
        _torch.Tensor.reshape: "reshape",
        _torch.Tensor.expand: "expand",
        _torch.Tensor.dim: "dim",
        _operator.add: "add",
        _operator.sub: "sub",
        _operator.mul: "mul",
        _operator.truediv: "truediv",
        _torch.Tensor.flatten: "flatten",
    }

    if target in func_map:
        return func_map[target]

    # Examples:
    # <built-in method conv2d of type object at 0x10816f290>
    # <function relu at 0x108fddbc0>
    # <function boolean_dispatch.<locals>.fn at 0x108fdc900>
    # <built-in function linear>
    # <class 'torchvision.models.googlenet.GoogLeNetOutputs'>
    tokens = str(target).split(" ")
    # Look for token after "method" or "function"
    token_keywords = (
        "method",
        "function",
        "<method",
        "<function",
        "<class",
    )
    for i in range(len(tokens)):
        token = tokens[i]
        if (token in token_keywords) and len(tokens) > i + 1:
            func_name = tokens[i + 1]
            if func_name[-1] == ">":
                func_name = func_name[:-1]
            return func_name
    return ""


class NodeSummary:
    """
    A data structure for nodes that contains the following information:
    node name, op kind, node shape, param count and param size in MB.
    """

    def __init__(
        self,
        node_name: str,
        op_kind: str,
        node_shape: _List[int],
        param_count: int,
        param_size: float,
    ):
        self.node_name = node_name
        self.op_kind = op_kind
        self.node_shape = node_shape
        self.param_count = param_count
        self.param_size = param_size


class ModelSummary:
    """
    Data structure that represents a summary of each node/module/function/attribute
    of some model. Contains a list of NodeSummary instances.
    """

    def __init__(self):
        self.nodes = list()

    def append(
        self,
        node_name: str,
        op_kind: str,
        node_shape: _List[int],
        param_count: int,
        param_size: float,
    ):
        node_summary = NodeSummary(
            node_name, op_kind, node_shape, param_count, param_size
        )
        self.nodes.append(node_summary)

    def __str__(self) -> str:
        from tabulate import tabulate as _tabulate
        fmt_table = [
            [
                node_summary.node_name,
                node_summary.op_kind,
                node_summary.node_shape,
                "{:,}".format(node_summary.param_count)
                if node_summary.param_count != 0
                else "--",
                "{:0.3f}".format(node_summary.param_size)
                if node_summary.param_size != 0.0
                else "--",
            ]
            for node_summary in self.nodes
        ]
        return str(
            _tabulate(
                fmt_table,
                headers=["name", "kind", "shape", "param #", "param size (MB)"],
            )
        )


def model_summary(
    model: _torch.nn.Module,
    traced_model: _torch.fx.GraphModule,
    input_shape: _Tuple[int],
) -> ModelSummary:
    """
    Method that takes in a model, traced model and expected input shape, and
    returns a ModelSummary data structure that summarizes the model.
    Note that this method will add a total node entry at the end of the ModelSummary.
    """

    sample_input = _torch.randn(input_shape)
    count_params = count_model_params(model)
    if count_params > 0 and next(model.parameters()).is_cuda:
        sample_input = sample_input.cuda()
    _torch.fx.passes.shape_prop.ShapeProp(traced_model).propagate(sample_input)
    modules = dict(model.named_modules())

    summary = ModelSummary()
    total_param_count = 0
    total_param_size = 0.0
    for node in traced_model.graph.nodes:
        op_kind = None
        if node.op == "placeholder":
            op_kind = "Input"
        elif node.op == "output":
            op_kind = "Output"
        elif node.op == "get_attr":
            op_kind = "Attr"
        elif node.op == "call_module":
            op_kind = modules[node.target].__class__.__name__
        elif node.op == "call_function":
            op_kind = parse_call_function_target(node.target)
        else:
            assert node.op == "call_method", "unsupported node op"
            op_kind = node.target

        param_count = 0
        param_size = 0
        if node.op == "call_module":
            for param_name, param in modules[node.target].named_parameters():
                param_count += volume(param.shape)
                param_size += volume(param.shape) * num_bytes(param.dtype)
        elif node.op == "call_function":
            weight_node = None
            bias_node = None
            if node.target in _LINEAR_CONV_DECONV_FUNCS:
                weight_node = node.args[1]
                bias_node = _fetch_argument(node.args, node.kwargs, None, 2, "bias")
            elif node.target == _torch.nn.functional.batch_norm:
                weight_node = _fetch_argument(node.args, node.kwargs, None, 3, "weight")
                bias_node = _fetch_argument(node.args, node.kwargs, None, 4, "bias")
            elif node.target == _torch.nn.functional.layer_norm:
                weight_node = _fetch_argument(node.args, node.kwargs, None, 2, "weight")
                bias_node = _fetch_argument(node.args, node.kwargs, None, 3, "bias")

            # If weight_node is None, then weight will be None
            # If bias_node is None, then bias will be None
            (weight, bias) = _fetch_func_params(model, weight_node, bias_node)
            if weight is not None:
                param_count += volume(weight.shape)
                param_size += volume(weight.shape) * num_bytes(weight.dtype)
            if bias is not None:
                param_count += volume(bias.shape)
                param_size += volume(bias.shape) * num_bytes(weight.dtype)

        param_size = param_size / 1e6  # 1 MB is 1000 * 1000 bytes

        node_shape = None
        if "tensor_meta" in node.meta:
            if not isinstance(
                node.meta["tensor_meta"], _torch.fx.passes.shape_prop.TensorMetadata
            ):
                # Multiple outputs. Simply use the first output shape
                node_shape = list(node.meta["tensor_meta"][0].shape)
            else:
                node_shape = list(node.meta["tensor_meta"].shape)

        summary.append(node.name, op_kind, node_shape, param_count, param_size)

        total_param_count += param_count
        total_param_size += param_size

    summary.append("Total", None, None, total_param_count, total_param_size)
    return summary


class Rewriter:
    """
    Graph utility which takes in the expected input shape to a model and the model,
    and rewrites the model so that relevant functions are mapped to the
    corresponding module operator types. Will not rewrite the model in-place.
    """

    def __init__(
        self,
        input_shape: _Tuple[int],
        model: _torch.nn.Module,
        traced_model: _torch.fx.GraphModule = None,
    ):
        self.input_shape = input_shape
        self.model = model

        for layer in self.model.modules():
            if hasattr(layer, "inplace"):
                layer.inplace = False

        if traced_model is None:
            self.graph_module = _torch.fx.symbolic_trace(self.model)
        else:
            self.graph_module = traced_model
        summary = model_summary(self.model, self.graph_module, self.input_shape)
        self.shapes = dict()
        # No need to process total entry at the end of the table
        for i in range(len(summary.nodes) - 1):
            node_summary = summary.nodes[i]
            self.shapes[node_summary.node_name] = node_summary.node_shape
        self.env = generate_env(self.model, self.graph_module, self.input_shape)

    def _replace_node_module(self, node: _torch.fx.Node, repl: _torch.nn.Module):
        """
        Method which takes in an input node and replacement module, and properly
        replaces the input node in the graph with a new node created for the
        replacement module. Preserves the node name. Intended to be a subgraph
        replacement with no global effect on the rest of the graph.
        """

        with self.graph_module.graph.inserting_after(node):
            # TODO: Check if node_name is correct for call_module function
            # Looks like "qualified" name needed here for the target parameter
            # https://pytorch.org/docs/stable/fx.html#torch.fx.Graph.call_module
            node_name = node.name
            self.model.__setattr__(node_name, repl)
            self.graph_module.__setattr__(node_name, repl)
            new_args = node.args if node.op == "call_module" else (node.args[0],)
            new_node = self.graph_module.graph.call_module(node_name, args=new_args)
            node.replace_all_uses_with(new_node)
            self.graph_module.graph.erase_node(node)
            new_node.name = node_name

    def _replace_node_function(
        self,
        node: _torch.fx.Node,
        target: _Any,
        fun_args: _Tuple[_Any],
        fun_kwargs: _Dict[str, _Any],
    ):
        """
        Method which takes in an input node and replacement target function, and properly
        replaces the input node in the graph with a new node created for the
        replacement function. Preserves the node name. Intended to be a subgraph
        replacement with no global effect on the rest of the graph.
        """

        with self.graph_module.graph.inserting_after(node):
            node_name = node.name
            new_node = self.graph_module.graph.call_function(
                target, args=fun_args, kwargs=fun_kwargs
            )
            node.replace_all_uses_with(new_node)
            self.graph_module.graph.erase_node(node)
            new_node.name = node_name

    def rewrite(self, debug: bool = False, custom_registry: _BaseRegistry = None):
        """
        Method which goes through each node in the graph and then each transform
        in the registry. If the transform pattern matches against the node, will
        compute the replacement module and replace the node with this module in
        the graph. This method then goes through the modified graph to delete
        unused operators and dead code. Finally, it recompiles and sanity-checks the
        graph before preparing the rewritten model.
        There is a debug flag which enables the printing of detailed information about
        the rewritten model and modified graph.
        """

        modules = dict(self.model.named_modules())
        registry = (
            _TransformRegistry.get_registry_values()
            if custom_registry is None
            else custom_registry
        )

        # fx represents its graph as an ordered list of
        # nodes so we can iterate through them.
        for node in self.graph_module.graph.nodes:
            for transform in registry:
                if transform.match_pattern(node, modules, self.env):
                    repl = transform.get_replacement(
                        self.model, node, self.shapes[node.name], modules, self.env
                    )
                    if transform.get_replacement_type() is _ReplacementType.kMODULE:
                        self._replace_node_module(node, repl)
                    elif transform.get_replacement_type() is _ReplacementType.kFUNCTION:
                        self._replace_node_function(node, repl[0], repl[1], repl[2])
                    else:
                        raise Exception("Replacement type is not supported")
                    # Regenerate modules
                    modules = dict(self.model.named_modules())

        self.graph_module.delete_all_unused_submodules()
        self.graph_module.graph.eliminate_dead_code()
        self.graph_module.recompile()
        # Does some checks to make sure the graph is well-formed.
        self.graph_module.graph.lint()
        self.model = _torch.fx.GraphModule(self.model, self.graph_module.graph)

        if debug:
            print("=======================================================")
            print("Printing human-readable graph module")
            self.graph_module.print_readable()
            print("=======================================================")
            print("Printing graph from graph module")
            print(self.graph_module.graph)
            print("=======================================================")
            print("Printing tabular graph")
            self.graph_module.graph.print_tabular()
            print("=======================================================")
            print("Printing named modules")
            for name, _ in self.model.named_modules():
                print("Found module with name", name)
            print("=======================================================")
