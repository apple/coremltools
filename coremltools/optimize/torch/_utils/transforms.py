#  Copyright (c) 2024, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import operator as _operator
from abc import ABC as _ABC
from abc import abstractmethod as _abstractmethod
from enum import Enum as _Enum
from typing import Any as _Any
from typing import Dict as _Dict
from typing import List as _List
from typing import Tuple as _Tuple

import torch as _torch

from coremltools.optimize.torch._utils.registry import BaseRegistry as _BaseRegistry
from coremltools._deps import _HAS_TORCH_VISION, MSG_TORCH_VISION_NOT_FOUND

if _HAS_TORCH_VISION:
    import torchvision as _torchvision

_CONV_FUNCS = (
    _torch.nn.functional.conv1d,
    _torch.nn.functional.conv2d,
    _torch.nn.functional.conv3d,
)

_DECONV_FUNCS = (
    _torch.nn.functional.conv_transpose1d,
    _torch.nn.functional.conv_transpose2d,
    _torch.nn.functional.conv_transpose3d,
)

_MAXPOOL_FUNCS = (
    _torch.nn.functional.max_pool1d,
    _torch.nn.functional.max_pool2d,
    _torch.nn.functional.max_pool3d,
)

_AVGPOOL_FUNCS = (
    _torch.nn.functional.avg_pool1d,
    _torch.nn.functional.avg_pool2d,
    _torch.nn.functional.avg_pool3d,
)

_ADAPTIVEAVGPOOL_FUNCS = (
    _torch.nn.functional.adaptive_avg_pool1d,
    _torch.nn.functional.adaptive_avg_pool2d,
    _torch.nn.functional.adaptive_avg_pool3d,
)

_LINEAR_CONV_DECONV_FUNCS = (_torch.nn.functional.linear,) + _CONV_FUNCS + _DECONV_FUNCS


def get_kernel_size(weights: _torch.nn.parameter.Parameter) -> _Tuple[int]:
    """
    Returns the kernel size of the input parameters given weights tensor.
    Assumes the first and second dimension correspond to channel dimensions.
    Assumes the remaining dimensions are spatial dimensions, which correspond
    to the kernel size.
    """

    num_spatial_dims = len(weights.shape) - 2
    assert num_spatial_dims >= 1, "Number of spatial dimensions should be at least 1"
    return tuple(weights.shape[2:])


def fetch_argument(
    args: _List[_Any],
    kwargs: _Dict[str, _Any],
    default: _Any,
    position: int,
    keyword: str = None,
) -> _Any:
    """
    Given a list of arguments, a dictionary of keyword arguments, a default value
    for the parameter if it cannot be found, the expected position of the parameter
    and the keyword corresponding to the parameter, this function determines the
    value of the parameter and returns it.
    """

    ret = default
    if keyword in kwargs:
        ret = kwargs[keyword]
    elif len(args) > position:
        ret = args[position]
    return ret


def load_arg(a, env):
    """
    Loads argument a from environment dictionary env and returns the result.
    """
    return _torch.fx.graph.map_arg(a, lambda n: env[n.name])


def fetch_attr(model: _torch.nn.Module, target: str) -> _Any:
    """
    Returns attribute within model that corresponds to the input target string.
    """

    target_atoms = target.split(".")
    attr_itr = model
    for i, atom in enumerate(target_atoms):
        if not hasattr(attr_itr, atom):
            raise RuntimeError(
                f"Node referenced nonexistent target {'.'.join(target_atoms[:i])}"
            )
        attr_itr = getattr(attr_itr, atom)
    return attr_itr


def fetch_func_params(
    model: _torch.nn.Module, weight_node: _torch.fx.Node, bias_node: _torch.fx.Node
) -> _Tuple[_Any]:
    """
    Given a model, weight node and bias node, this function fetches the attributes
    corresponding to these nodes and returns them as a 2-tuple (weight, bias)
    """

    weight = None
    if weight_node is not None:
        assert weight_node.op == "get_attr", "unsupported op for weight node"
        weight = fetch_attr(model, weight_node.target)
    bias = None
    if bias_node is not None:
        assert bias_node.op == "get_attr", "unsupported op for bias node"
        bias = fetch_attr(model, bias_node.target)
    return (weight, bias)


class ReplacementType(_Enum):
    kMODULE = 0
    kFUNCTION = 1


class Transform(_ABC):
    """
    Abstract base class for all transformations.
    """

    @_abstractmethod
    def __str__(self) -> str:
        """
        Returns the name of the transform.
        """
        pass

    @_abstractmethod
    def match_pattern(
        self,
        node: _torch.fx.Node,
        modules: _Dict[str, _torch.nn.Module],
        env: _Dict[str, _Any],
    ) -> bool:
        """
        Determines whether the input node matches the requirements for the transform
        to take place and returns the result.
        """
        pass

    @_abstractmethod
    def get_replacement_type(self) -> ReplacementType:
        """
        Returns the replacement type enumeration for this transformation.
        """
        pass

    @_abstractmethod
    def get_replacement(
        self,
        node: _torch.fx.Node,
        output_shape: _Tuple[int],
        modules: _Dict[str, _torch.nn.Module],
        env: _Dict[str, _Any],
    ) -> _Any:
        """
        Computes the transformed node as a result of the transformation given
        input node and expected output shape of this node.
        """
        pass


# TODO: Add transforms for
# adaptive_max_pool1d, adaptive_max_pool2d, adaptive_max_pool3d
class TransformRegistry(_BaseRegistry):
    """
    Registry that contains general transforms for rewriting PyTorch network to canonical representation.
    """

    def __init_subclass__(cls, *args, **kwargs):
        TransformRegistry.instantiate(cls, *args, **kwargs)


def requires_grad(weights: _Any) -> bool:
    """
    If the given weights is a torch parameter instance and has requires_grad == True,
    returns True. Otherwise returns False.
    """
    return isinstance(weights, _torch.nn.parameter.Parameter) and weights.requires_grad


class FuncToModuleLinear(Transform, TransformRegistry):
    """
    Subclass of class Transform that rewrites Linear functions to Linear modules.
    """

    def __str__(self) -> str:
        return "FuncToModuleLinear"

    def match_pattern(
        self,
        node: _torch.fx.Node,
        modules: _Dict[str, _torch.nn.Module],
        env: _Dict[str, _Any],
    ) -> bool:
        # Checks if we are calling a function
        # The target attribute is the function that call_function calls
        return node.op == "call_function" and node.target == _torch.nn.functional.linear

    def get_replacement_type(self) -> ReplacementType:
        return ReplacementType.kMODULE

    def get_replacement(
        self,
        model: _torch.nn.Module,
        node: _torch.fx.Node,
        output_shape: _Tuple[int],
        modules: _Dict[str, _torch.nn.Module],
        env: _Dict[str, _Any],
    ) -> _torch.nn.Module:
        weight_node = node.args[1]
        bias_node = fetch_argument(node.args, node.kwargs, None, 2, "bias")
        (weight, bias) = fetch_func_params(model, weight_node, bias_node)

        out_features = weight.shape[0]
        in_features = weight.shape[1]
        repl = _torch.nn.Linear(in_features, out_features, bias is not None)
        repl.weight = _torch.nn.parameter.Parameter(
            weight, requires_grad=requires_grad(weight)
        )
        if bias is not None:
            repl.bias = _torch.nn.parameter.Parameter(
                bias, requires_grad=requires_grad(bias)
            )
        return repl


class FuncToModuleConv(Transform, TransformRegistry):
    """
    Subclass of class Transform that rewrites Conv functions to Conv modules.
    """

    def __str__(self) -> str:
        return "FuncToModuleConv"

    def match_pattern(
        self,
        node: _torch.fx.Node,
        modules: _Dict[str, _torch.nn.Module],
        env: _Dict[str, _Any],
    ) -> bool:
        # Checks if we are calling a function
        # The target attribute is the function that call_function calls
        return node.op == "call_function" and node.target in _CONV_FUNCS

    def get_replacement_type(self) -> ReplacementType:
        return ReplacementType.kMODULE

    def get_replacement(
        self,
        model: _torch.nn.Module,
        node: _torch.fx.Node,
        output_shape: _Tuple[int],
        modules: _Dict[str, _torch.nn.Module],
        env: _Dict[str, _Any],
    ) -> _torch.nn.Module:
        weight_node = node.args[1]
        bias_node = fetch_argument(node.args, node.kwargs, None, 2, "bias")
        (weight, bias) = fetch_func_params(model, weight_node, bias_node)

        kernel_size = get_kernel_size(weight)
        stride = fetch_argument(node.args, node.kwargs, 1, 3, "stride")
        padding = fetch_argument(node.args, node.kwargs, 0, 4, "padding")
        dilation = fetch_argument(node.args, node.kwargs, 1, 5, "dilation")
        groups = fetch_argument(node.args, node.kwargs, 1, 6, "groups")

        out_channels = weight.shape[0]
        in_channels = weight.shape[1] * groups

        args = [
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias is not None,
        ]

        repl = None
        if node.target == _torch.nn.functional.conv1d:
            repl = _torch.nn.Conv1d(*args)
        elif node.target == _torch.nn.functional.conv2d:
            repl = _torch.nn.Conv2d(*args)
        else:
            assert node.target == _torch.nn.functional.conv3d, "unsupported node target"
            repl = _torch.nn.Conv3d(*args)

        repl.weight = _torch.nn.parameter.Parameter(
            weight, requires_grad=requires_grad(weight)
        )
        if bias is not None:
            repl.bias = _torch.nn.parameter.Parameter(
                bias, requires_grad=requires_grad(bias)
            )
        return repl


class FuncToModuleDeconv(Transform, TransformRegistry):
    """
    Subclass of class Transform that rewrites Deconv functions to Deconv modules.
    """

    def __str__(self) -> str:
        return "FuncToModuleDeconv"

    def match_pattern(
        self,
        node: _torch.fx.Node,
        modules: _Dict[str, _torch.nn.Module],
        env: _Dict[str, _Any],
    ) -> bool:
        # Checks if we are calling a function
        # The target attribute is the function that call_function calls
        return node.op == "call_function" and node.target in _DECONV_FUNCS

    def get_replacement_type(self) -> ReplacementType:
        return ReplacementType.kMODULE

    def get_replacement(
        self,
        model: _torch.nn.Module,
        node: _torch.fx.Node,
        output_shape: _Tuple[int],
        modules: _Dict[str, _torch.nn.Module],
        env: _Dict[str, _Any],
    ) -> _torch.nn.Module:
        weight_node = node.args[1]
        bias_node = fetch_argument(node.args, node.kwargs, None, 2, "bias")
        (weight, bias) = fetch_func_params(model, weight_node, bias_node)

        kernel_size = get_kernel_size(weight)
        stride = fetch_argument(node.args, node.kwargs, 1, 3, "stride")
        padding = fetch_argument(node.args, node.kwargs, 0, 4, "padding")
        output_padding = fetch_argument(node.args, node.kwargs, 0, 5, "output_padding")
        groups = fetch_argument(node.args, node.kwargs, 1, 6, "groups")
        dilation = fetch_argument(node.args, node.kwargs, 1, 7, "dilation")

        out_channels = weight.shape[1]
        in_channels = weight.shape[0] * groups

        args = [
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding,
            groups,
            bias is not None,
            dilation,
        ]

        repl = None
        if node.target == _torch.nn.functional.conv_transpose1d:
            repl = _torch.nn.ConvTranspose1d(*args)
        elif node.target == _torch.nn.functional.conv_transpose2d:
            repl = _torch.nn.ConvTranspose2d(*args)
        else:
            assert (
                node.target == _torch.nn.functional.conv_transpose3d
            ), "unsupported node target"
            repl = _torch.nn.ConvTranspose3d(*args)

        repl.weight = _torch.nn.parameter.Parameter(
            weight, requires_grad=requires_grad(weight)
        )
        if bias is not None:
            repl.bias = _torch.nn.parameter.Parameter(
                bias, requires_grad=requires_grad(bias)
            )
        return repl


class FuncToModuleMaxPool(Transform, TransformRegistry):
    """
    Subclass of class Transform that rewrites MaxPool functions to MaxPool modules.
    """

    def __str__(self) -> str:
        return "FuncToModuleMaxPool"

    def match_pattern(
        self,
        node: _torch.fx.Node,
        modules: _Dict[str, _torch.nn.Module],
        env: _Dict[str, _Any],
    ) -> bool:
        # Checks if we are calling a function
        # The target attribute is the function that call_function calls
        return node.op == "call_function" and node.target in _MAXPOOL_FUNCS

    def get_replacement_type(self) -> ReplacementType:
        return ReplacementType.kMODULE

    def get_replacement(
        self,
        model: _torch.nn.Module,
        node: _torch.fx.Node,
        output_shape: _Tuple[int],
        modules: _Dict[str, _torch.nn.Module],
        env: _Dict[str, _Any],
    ) -> _torch.nn.Module:
        kernel_size = fetch_argument(node.args, node.kwargs, None, 1, "kernel_size")
        stride = fetch_argument(node.args, node.kwargs, None, 2, "stride")
        padding = fetch_argument(node.args, node.kwargs, 0, 3, "padding")
        dilation = fetch_argument(node.args, node.kwargs, 1, 4, "dilation")
        ceil_mode = fetch_argument(node.args, node.kwargs, False, 5, "ceil_mode")
        return_indices = fetch_argument(
            node.args, node.kwargs, False, 6, "return_indices"
        )

        args = [kernel_size, stride, padding, dilation, return_indices, ceil_mode]

        repl = None
        if node.target == _torch.nn.functional.max_pool1d:
            repl = _torch.nn.MaxPool1d(*args)
        elif node.target == _torch.nn.functional.max_pool2d:
            repl = _torch.nn.MaxPool2d(*args)
        else:
            assert (
                node.target == _torch.nn.functional.max_pool3d
            ), "unsupported node target"
            repl = _torch.nn.MaxPool3d(*args)
        return repl


class FuncToModuleAvgPool(Transform, TransformRegistry):
    """
    Subclass of class Transform that rewrites AvgPool functions to AvgPool modules.
    """

    def __str__(self) -> str:
        return "FuncToModuleAvgPool"

    def match_pattern(
        self,
        node: _torch.fx.Node,
        modules: _Dict[str, _torch.nn.Module],
        env: _Dict[str, _Any],
    ) -> bool:
        # Checks if we are calling a function
        # The target attribute is the function that call_function calls
        return node.op == "call_function" and node.target in _AVGPOOL_FUNCS

    def get_replacement_type(self) -> ReplacementType:
        return ReplacementType.kMODULE

    def get_replacement(
        self,
        model: _torch.nn.Module,
        node: _torch.fx.Node,
        output_shape: _Tuple[int],
        modules: _Dict[str, _torch.nn.Module],
        env: _Dict[str, _Any],
    ) -> _torch.nn.Module:
        kernel_size = fetch_argument(node.args, node.kwargs, None, 1, "kernel_size")
        stride = fetch_argument(node.args, node.kwargs, None, 2, "stride")
        padding = fetch_argument(node.args, node.kwargs, 0, 3, "padding")
        ceil_mode = fetch_argument(node.args, node.kwargs, False, 4, "ceil_mode")
        count_include_pad = fetch_argument(
            node.args, node.kwargs, True, 5, "count_include_pad"
        )

        args = [kernel_size, stride, padding, ceil_mode, count_include_pad]

        repl = None
        if node.target == _torch.nn.functional.avg_pool1d:
            repl = _torch.nn.AvgPool1d(*args)
        elif node.target == _torch.nn.functional.avg_pool2d:
            divisor_override = fetch_argument(
                node.args, node.kwargs, None, 6, "divisor_override"
            )
            repl = _torch.nn.AvgPool2d(*args, divisor_override)
        else:
            assert (
                node.target == _torch.nn.functional.avg_pool3d
            ), "unsupported node target"
            divisor_override = fetch_argument(
                node.args, node.kwargs, None, 6, "divisor_override"
            )
            repl = _torch.nn.AvgPool3d(*args, divisor_override)
        return repl


class FuncToModuleAdaptiveAvgPool(Transform, TransformRegistry):
    """
    Subclass of class Transform that rewrites AdaptiveAvgPool functions to
    AdaptiveAvgPool modules.
    """

    def __str__(self) -> str:
        return "FuncToModuleAdaptiveAvgPool"

    def match_pattern(
        self,
        node: _torch.fx.Node,
        modules: _Dict[str, _torch.nn.Module],
        env: _Dict[str, _Any],
    ) -> bool:
        # Checks if we are calling a function
        # The target attribute is the function that call_function calls
        return node.op == "call_function" and node.target in _ADAPTIVEAVGPOOL_FUNCS

    def get_replacement_type(self) -> ReplacementType:
        return ReplacementType.kMODULE

    def get_replacement(
        self,
        model: _torch.nn.Module,
        node: _torch.fx.Node,
        output_shape: _Tuple[int],
        modules: _Dict[str, _torch.nn.Module],
        env: _Dict[str, _Any],
    ) -> _torch.nn.Module:
        output_size = node.args[1]

        repl = None
        if node.target == _torch.nn.functional.adaptive_avg_pool1d:
            repl = _torch.nn.AdaptiveAvgPool1d(output_size)
        elif node.target == _torch.nn.functional.adaptive_avg_pool2d:
            repl = _torch.nn.AdaptiveAvgPool2d(output_size)
        else:
            assert (
                node.target == _torch.nn.functional.adaptive_avg_pool3d
            ), "unsupported node target"
            repl = _torch.nn.AdaptiveAvgPool3d(output_size)

        return repl


class FuncToModuleDropout(Transform, TransformRegistry):
    """
    Subclass of class Transform that rewrites Dropout functions to Dropout modules.
    """

    def __str__(self) -> str:
        return "FuncToModuleDropout"

    def match_pattern(
        self,
        node: _torch.fx.Node,
        modules: _Dict[str, _torch.nn.Module],
        env: _Dict[str, _Any],
    ) -> bool:
        # Checks if we are calling a function
        # The target attribute is the function that call_function calls
        return node.op == "call_function" and (
            node.target == _torch.nn.functional.dropout
            or node.target == _torch.nn.functional.dropout1d
            or node.target == _torch.nn.functional.dropout2d
            or node.target == _torch.nn.functional.dropout3d
        )

    def get_replacement_type(self) -> ReplacementType:
        return ReplacementType.kMODULE

    def get_replacement(
        self,
        model: _torch.nn.Module,
        node: _torch.fx.Node,
        output_shape: _Tuple[int],
        modules: _Dict[str, _torch.nn.Module],
        env: _Dict[str, _Any],
    ) -> _torch.nn.Module:
        p = fetch_argument(node.args, node.kwargs, 0.5, 1, "p")
        inplace = fetch_argument(node.args, node.kwargs, False, 3, "inplace")

        args = [p, inplace]

        repl = None
        if node.target == _torch.nn.functional.dropout:
            repl = _torch.nn.Dropout(*args)
        elif node.target == _torch.nn.functional.dropout1d:
            repl = _torch.nn.Dropout1d(*args)
        elif node.target == _torch.nn.functional.dropout2d:
            repl = _torch.nn.Dropout2d(*args)
        else:
            assert node.target == _torch.nn.functional.dropout3d
            repl = _torch.nn.Dropout3d(*args)

        return repl


class FuncToModuleFlatten(Transform, TransformRegistry):
    """
    Subclass of class Transform that rewrites Flatten functions to Flatten modules.
    """

    def __str__(self) -> str:
        return "FuncToModuleFlatten"

    def match_pattern(
        self,
        node: _torch.fx.Node,
        modules: _Dict[str, _torch.nn.Module],
        env: _Dict[str, _Any],
    ) -> bool:
        # Checks if we are calling a function
        # The target attribute is the function that call_function calls
        return node.op == "call_function" and node.target == _torch.flatten

    def get_replacement_type(self) -> ReplacementType:
        return ReplacementType.kMODULE

    def get_replacement(
        self,
        model: _torch.nn.Module,
        node: _torch.fx.Node,
        output_shape: _Tuple[int],
        modules: _Dict[str, _torch.nn.Module],
        env: _Dict[str, _Any],
    ) -> _torch.nn.Module:
        start_dim = fetch_argument(node.args, node.kwargs, 0, 1, "start_dim")
        end_dim = fetch_argument(node.args, node.kwargs, -1, 2, "end_dim")
        repl = _torch.nn.Flatten(start_dim, end_dim)
        return repl


class MethodToModuleFlatten(Transform, TransformRegistry):
    """
    Subclass of class Transform that rewrites Flatten methods to Flatten modules.
    """

    def __str__(self) -> str:
        return "MethodToModuleFlatten"

    def match_pattern(
        self,
        node: _torch.fx.Node,
        modules: _Dict[str, _torch.nn.Module],
        env: _Dict[str, _Any],
    ) -> bool:
        # Checks if we are calling a method
        if node.op == "call_method":
            self_obj, *args = load_arg(node.args, env)
            return isinstance(self_obj, _torch.Tensor) and node.target == "flatten"
        else:
            return False

    def get_replacement_type(self) -> ReplacementType:
        return ReplacementType.kMODULE

    def get_replacement(
        self,
        model: _torch.nn.Module,
        node: _torch.fx.Node,
        output_shape: _Tuple[int],
        modules: _Dict[str, _torch.nn.Module],
        env: _Dict[str, _Any],
    ) -> _torch.nn.Module:
        start_dim = fetch_argument(node.args, node.kwargs, 0, 1, "start_dim")
        end_dim = fetch_argument(node.args, node.kwargs, -1, 2, "end_dim")
        repl = _torch.nn.Flatten(start_dim, end_dim)
        return repl


class FuncToModuleBatchNorm(Transform, TransformRegistry):
    """
    Subclass of class Transform that rewrites BatchNorm functions to BatchNorm modules.
    """

    def __str__(self) -> str:
        return "FuncToModuleBatchNorm"

    def match_pattern(
        self,
        node: _torch.fx.Node,
        modules: _Dict[str, _torch.nn.Module],
        env: _Dict[str, _Any],
    ) -> bool:
        # Checks if we are calling a function
        # The target attribute is the function that call_function calls
        return (
            node.op == "call_function"
            and node.target == _torch.nn.functional.batch_norm
        )

    def get_replacement_type(self) -> ReplacementType:
        return ReplacementType.kMODULE

    def get_replacement(
        self,
        model: _torch.nn.Module,
        node: _torch.fx.Node,
        output_shape: _Tuple[int],
        modules: _Dict[str, _torch.nn.Module],
        env: _Dict[str, _Any],
    ) -> _torch.nn.Module:
        running_mean_node = node.args[1]
        running_var_node = node.args[2]
        weight_node = fetch_argument(node.args, node.kwargs, None, 3, "weight")
        bias_node = fetch_argument(node.args, node.kwargs, None, 4, "bias")
        # Does not look like this is required by the module
        training = fetch_argument(node.args, node.kwargs, False, 5, "training")
        momentum = fetch_argument(node.args, node.kwargs, 0.1, 6, "momentum")
        eps = fetch_argument(node.args, node.kwargs, 1e-05, 7, "eps")

        running_mean = None
        assert (
            running_mean_node.op == "get_attr"
        ), "unsupported op for running mean node"
        running_mean = fetch_attr(model, running_mean_node.target)

        running_var = None
        assert running_var_node.op == "get_attr", "unsupported op for running var node"
        running_var = fetch_attr(model, running_var_node.target)

        (weight, bias) = fetch_func_params(model, weight_node, bias_node)

        num_spatial_dims = len(output_shape) - 2
        num_features = running_mean.shape[0]
        affine = weight is not None or bias is not None
        # The function batch_norm always tracks the running stats
        track_running_stats = True
        args = [num_features, eps, momentum, affine, track_running_stats]

        repl = None
        if num_spatial_dims == 0 or num_spatial_dims == 1:
            repl = _torch.nn.BatchNorm1d(*args)
        elif num_spatial_dims == 2:
            repl = _torch.nn.BatchNorm2d(*args)
        elif num_spatial_dims == 3:
            repl = _torch.nn.BatchNorm3d(*args)
        else:
            raise ValueError("Unsupported number of spatial dimensions for batch norm")

        # Note that running_mean and running_var are not trainable parameters
        if running_mean is not None:
            if isinstance(running_mean, _torch.nn.parameter.Parameter):
                repl.running_mean = _torch.nn.parameter.Parameter(
                    running_mean, requires_grad=False
                )
            else:
                repl.register_buffer("running_mean", running_mean)
        if running_var is not None:
            if isinstance(running_var, _torch.nn.parameter.Parameter):
                repl.running_var = _torch.nn.parameter.Parameter(
                    running_var, requires_grad=False
                )
            else:
                repl.register_buffer("running_var", running_var)
        if weight is not None:
            repl.weight = _torch.nn.parameter.Parameter(
                weight, requires_grad=requires_grad(weight)
            )
        if bias is not None:
            repl.bias = _torch.nn.parameter.Parameter(
                bias, requires_grad=requires_grad(bias)
            )

        return repl


class FuncToModuleLayerNorm(Transform, TransformRegistry):
    """
    Subclass of class Transform that rewrites LayerNorm functions to LayerNorm modules.
    """

    def __str__(self) -> str:
        return "FuncToModuleLayerNorm"

    def match_pattern(
        self,
        node: _torch.fx.Node,
        modules: _Dict[str, _torch.nn.Module],
        env: _Dict[str, _Any],
    ) -> bool:
        # Checks if we are calling a function
        # The target attribute is the function that call_function calls
        return (
            node.op == "call_function"
            and node.target == _torch.nn.functional.layer_norm
        )

    def get_replacement_type(self) -> ReplacementType:
        return ReplacementType.kMODULE

    def get_replacement(
        self,
        model: _torch.nn.Module,
        node: _torch.fx.Node,
        output_shape: _Tuple[int],
        modules: _Dict[str, _torch.nn.Module],
        env: _Dict[str, _Any],
    ) -> _torch.nn.Module:
        normalized_shape = node.args[1]
        weight_node = fetch_argument(node.args, node.kwargs, None, 2, "weight")
        bias_node = fetch_argument(node.args, node.kwargs, None, 3, "bias")
        eps = fetch_argument(node.args, node.kwargs, 1e-05, 4, "eps")

        (weight, bias) = fetch_func_params(model, weight_node, bias_node)

        elementwise_affine = weight is not None
        args = [tuple(normalized_shape), eps, elementwise_affine]
        repl = _torch.nn.LayerNorm(*args)
        if elementwise_affine:
            repl.weight = _torch.nn.parameter.Parameter(
                weight, requires_grad=requires_grad(weight)
            )
            if bias is not None:
                repl.bias = _torch.nn.parameter.Parameter(
                    bias, requires_grad=requires_grad(bias)
                )

        return repl


class FuncToModuleRelu(Transform, TransformRegistry):
    """
    Subclass of class Transform that rewrites Relu functions to Relu modules.
    """

    def __str__(self) -> str:
        return "FuncToModuleRelu"

    def match_pattern(
        self,
        node: _torch.fx.Node,
        modules: _Dict[str, _torch.nn.Module],
        env: _Dict[str, _Any],
    ) -> bool:
        # Checks if we are calling a function
        # The target attribute is the function that call_function calls
        return node.op == "call_function" and node.target == _torch.nn.functional.relu

    def get_replacement_type(self) -> ReplacementType:
        return ReplacementType.kMODULE

    def get_replacement(
        self,
        model: _torch.nn.Module,
        node: _torch.fx.Node,
        output_shape: _Tuple[int],
        modules: _Dict[str, _torch.nn.Module],
        env: _Dict[str, _Any],
    ) -> _torch.nn.Module:
        # inplace == False by default
        repl = _torch.nn.ReLU()
        return repl


class FuncToModuleSigmoid(Transform, TransformRegistry):
    """
    Subclass of class Transform that rewrites Sigmoid functions to Sigmoid modules.
    """

    def __str__(self) -> str:
        return "FuncToModuleSigmoid"

    def match_pattern(
        self,
        node: _torch.fx.Node,
        modules: _Dict[str, _torch.nn.Module],
        env: _Dict[str, _Any],
    ) -> bool:
        # Checks if we are calling a function
        # The target attribute is the function that call_function calls
        return node.op == "call_function" and node.target == _torch.sigmoid

    def get_replacement_type(self) -> ReplacementType:
        return ReplacementType.kMODULE

    def get_replacement(
        self,
        model: _torch.nn.Module,
        node: _torch.fx.Node,
        output_shape: _Tuple[int],
        modules: _Dict[str, _torch.nn.Module],
        env: _Dict[str, _Any],
    ) -> _torch.nn.Module:
        repl = _torch.nn.Sigmoid()
        return repl


class FuncToModuleSoftmax(Transform, TransformRegistry):
    """
    Subclass of class Transform that rewrites Softmax functions to Softmax modules.
    """

    def __str__(self) -> str:
        return "FuncToModuleSoftmax"

    def match_pattern(
        self,
        node: _torch.fx.Node,
        modules: _Dict[str, _torch.nn.Module],
        env: _Dict[str, _Any],
    ) -> bool:
        # Checks if we are calling a function
        # The target attribute is the function that call_function calls
        return (
            node.op == "call_function" and node.target == _torch.nn.functional.softmax
        )

    def get_replacement_type(self) -> ReplacementType:
        return ReplacementType.kMODULE

    def get_replacement(
        self,
        model: _torch.nn.Module,
        node: _torch.fx.Node,
        output_shape: _Tuple[int],
        modules: _Dict[str, _torch.nn.Module],
        env: _Dict[str, _Any],
    ) -> _torch.nn.Module:
        dim = fetch_argument(node.args, node.kwargs, None, 1, "dim")
        dtype = fetch_argument(node.args, node.kwargs, None, 3, "dtype")
        assert dtype is None
        repl = _torch.nn.Softmax(dim)
        return repl


class MethodToFuncPermute(Transform, TransformRegistry):
    """
    Subclass of class Transform that rewrites Permute methods to Permute functions.
    """

    def __str__(self) -> str:
        return "MethodToFuncPermute"

    def match_pattern(
        self,
        node: _torch.fx.Node,
        modules: _Dict[str, _torch.nn.Module],
        env: _Dict[str, _Any],
    ) -> bool:
        # Checks if we are calling a method
        if node.op == "call_method":
            self_obj, *args = load_arg(node.args, env)
            tensor_check = isinstance(self_obj, _torch.Tensor)

            if not _HAS_TORCH_VISION:
                raise ImportError(MSG_TORCH_VISION_NOT_FOUND)

            torchvision_check = self_obj == _torchvision.ops.misc.Permute
            return (tensor_check or torchvision_check) and node.target == "permute"
        else:
            return False

    def get_replacement_type(self) -> ReplacementType:
        return ReplacementType.kFUNCTION

    def get_replacement(
        self,
        model: _torch.nn.Module,
        node: _torch.fx.Node,
        output_shape: _Tuple[int],
        modules: _Dict[str, _torch.nn.Module],
        env: _Dict[str, _Any],
    ) -> _Any:
        input = node.args[0]
        # dims must be a tuple of int
        dims = None
        if isinstance(node.args[1], int):
            dims = tuple(node.args[1:])
        else:
            dims = tuple(node.args[1])

        target = _torch.permute
        fun_args = (input, dims)
        fun_kwargs = dict()

        return target, fun_args, fun_kwargs


class MethodToFuncReshape(Transform, TransformRegistry):
    """
    Subclass of class Transform that rewrites Reshape/View methods to Reshape functions.
    """

    def __str__(self) -> str:
        return "MethodToFuncReshape"

    def match_pattern(
        self,
        node: _torch.fx.Node,
        modules: _Dict[str, _torch.nn.Module],
        env: _Dict[str, _Any],
    ) -> bool:
        # Checks if we are calling a method
        if node.op == "call_method":
            self_obj, *args = load_arg(node.args, env)
            return isinstance(self_obj, _torch.Tensor) and (
                node.target == "reshape" or node.target == "view"
            )
        else:
            return False

    def get_replacement_type(self) -> ReplacementType:
        return ReplacementType.kFUNCTION

    def get_replacement(
        self,
        model: _torch.nn.Module,
        node: _torch.fx.Node,
        output_shape: _Tuple[int],
        modules: _Dict[str, _torch.nn.Module],
        env: _Dict[str, _Any],
    ) -> _Any:
        input = node.args[0]
        # shape must be a tuple of int
        shape = None

        any_nodes = False
        for item in node.args[1:]:
            if isinstance(item, _torch.fx.Node):
                any_nodes = True

        all_ints = True
        for item in node.args[1:]:
            if not isinstance(item, int):
                all_ints = False
                break

        if any_nodes:
            _, *args = load_arg(node.args, env)
            shape = tuple(args)
        elif all_ints:
            shape = tuple(node.args[1:])
        elif isinstance(node.args[1], _torch.Size) or isinstance(node.args[1], tuple):
            shape = tuple(node.args[1])
        else:
            raise Exception(
                "Invalid view/reshape. Expected input arguments to be singleton Size, all ints or tuple of ints."
            )

        target = _torch.reshape
        fun_args = (input, shape)
        fun_kwargs = dict()

        return target, fun_args, fun_kwargs


class MethodToFuncTranspose(Transform, TransformRegistry):
    """
    Subclass of class Transform that rewrites Transpose methods to Transpose functions.
    """

    def __str__(self) -> str:
        return "MethodToFuncTranspose"

    def match_pattern(
        self,
        node: _torch.fx.Node,
        modules: _Dict[str, _torch.nn.Module],
        env: _Dict[str, _Any],
    ) -> bool:
        # Checks if we are calling a method
        if node.op == "call_method":
            self_obj, *args = load_arg(node.args, env)
            return isinstance(self_obj, _torch.Tensor) and node.target == "transpose"
        else:
            return False

    def get_replacement_type(self) -> ReplacementType:
        return ReplacementType.kFUNCTION

    def get_replacement(
        self,
        model: _torch.nn.Module,
        node: _torch.fx.Node,
        output_shape: _Tuple[int],
        modules: _Dict[str, _torch.nn.Module],
        env: _Dict[str, _Any],
    ) -> _Any:
        input = node.args[0]
        dim0 = node.args[1]
        dim1 = node.args[2]

        target = _torch.transpose
        fun_args = (input, dim0, dim1)
        fun_kwargs = dict()

        return target, fun_args, fun_kwargs


class MethodToFuncChunk(Transform, TransformRegistry):
    """
    Subclass of class Transform that rewrites Chunk methods to Chunk functions.
    """

    def __str__(self) -> str:
        return "MethodToFuncChunk"

    def match_pattern(
        self,
        node: _torch.fx.Node,
        modules: _Dict[str, _torch.nn.Module],
        env: _Dict[str, _Any],
    ) -> bool:
        # Checks if we are calling a method
        if node.op == "call_method":
            self_obj, *args = load_arg(node.args, env)
            return isinstance(self_obj, _torch.Tensor) and node.target == "chunk"
        else:
            return False

    def get_replacement_type(self) -> ReplacementType:
        return ReplacementType.kFUNCTION

    def get_replacement(
        self,
        model: _torch.nn.Module,
        node: _torch.fx.Node,
        output_shape: _Tuple[int],
        modules: _Dict[str, _torch.nn.Module],
        env: _Dict[str, _Any],
    ) -> _Any:
        input = node.args[0]
        chunks = node.args[1]
        dim = fetch_argument(node.args, node.kwargs, 0, 2, "dim")

        target = _torch.chunk
        fun_args = (input, chunks, dim)
        fun_kwargs = dict()

        return target, fun_args, fun_kwargs


class FuncToFuncOperator(Transform, TransformRegistry):
    """
    Subclass of class Transform that rewrites pointwise operator functions to
    equivalent pointwise PyTorch operator functions.
    Purpose of this transform is to replace operators with the equivalent PyTorch
    operators. This protects against backward hook failures due to inplace operations
    such as +=, -=, *= and /=.
    """

    repl_map = {
        _operator.add: _torch.add,
        _operator.sub: _torch.sub,
        _operator.mul: _torch.mul,
        _operator.truediv: _torch.div,
    }

    def __str__(self) -> str:
        return "FuncToFuncOperator"

    def match_pattern(
        self,
        node: _torch.fx.Node,
        modules: _Dict[str, _torch.nn.Module],
        env: _Dict[str, _Any],
    ) -> bool:
        # Checks if we are calling a function
        # The target attribute is the function that call_function calls
        return node.op == "call_function" and node.target in FuncToFuncOperator.repl_map

    def get_replacement_type(self) -> ReplacementType:
        return ReplacementType.kFUNCTION

    def get_replacement(
        self,
        model: _torch.nn.Module,
        node: _torch.fx.Node,
        output_shape: _Tuple[int],
        modules: _Dict[str, _torch.nn.Module],
        env: _Dict[str, _Any],
    ) -> _Any:
        target = FuncToFuncOperator.repl_map[node.target]
        fun_args = node.args
        fun_kwargs = node.kwargs

        return target, fun_args, fun_kwargs
