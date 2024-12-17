#  Copyright (c) 2024, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import operator as _operator
from typing import Callable as _Callable
from typing import List as _List
from typing import Optional as _Optional

import torch as _torch
from torch.ao.quantization.quantizer.quantizer import Quantizer as _TorchQuantizer
from torch.ao.quantization.quantizer.xnnpack_quantizer import _get_module_name_filter
from torch.fx import Node as _Node

import coremltools.optimize.torch.quantization._coreml_quantizer_utils as _annotation_utils
from coremltools.optimize.torch._utils.python_utils import FunctionRegistryMixin as _FunctionRegistryMixin
from coremltools.optimize.torch.quantization._annotation_config import (
    AnnotationConfig as _AnnotationConfig,
)
from coremltools.optimize.torch.quantization.quantization_config import (
    LinearQuantizerConfig as _LinearQuantizerConfig,
)
from coremltools.optimize.torch.quantization.quantization_config import (
    ModuleLinearQuantizerConfig as _ModuleLinearQuantizerConfig,
)


class _AnnotationPatternRegistry(_FunctionRegistryMixin):
    """
    A registry of quantization annotation rules.
    """
    @classmethod
    def get_annotators(cls):
        return cls.REGISTRY


@_AnnotationPatternRegistry.register("conv_act")
def _annotate_conv_act(
    model: _torch.fx.GraphModule,
    quantization_config: _Optional[_AnnotationConfig],
    filter_fn: _Optional[_Callable[[_Node], bool]] = None,
) -> _Optional[_List[_List[_Node]]]:
    """
    Annotates input -> conv -> activation -> output
    """
    return _annotation_utils.annotate_conv_bn_act_helper(
        model, quantization_config, filter_fn, use_bn=False
    )


@_AnnotationPatternRegistry.register("conv_bn_act")
def _annotate_conv_bn_act(
    model: _torch.fx.GraphModule,
    quantization_config: _Optional[_AnnotationConfig],
    filter_fn: _Optional[_Callable[[_Node], bool]] = None,
) -> _Optional[_List[_List[_Node]]]:
    """
    Annotates input -> conv -> batch_norm -> activation -> output
    """
    return _annotation_utils.annotate_conv_bn_act_helper(
        model, quantization_config, filter_fn, use_bn=True
    )


@_AnnotationPatternRegistry.register("conv_bn")
def _annotate_conv_bn(
    model: _torch.fx.GraphModule,
    quantization_config: _Optional[_AnnotationConfig],
    filter_fn: _Optional[_Callable[[_Node], bool]] = None,
) -> _Optional[_List[_List[_Node]]]:
    """
    Annotates input -> conv -> batch_norm -> output
    """
    annotated_partitions = []

    conv_dims = [1, 2, 3]
    for conv_dim in conv_dims:
        pattern_gm = _annotation_utils.get_conv_bn_pattern(
            conv_dim, act_fn=None, act_in_place=False
        )
        annotated_partitions.extend(
            _annotation_utils.annotate_weighted_mod_pattern(
                model, pattern_gm, quantization_config, filter_fn
            )
        )
    return annotated_partitions


@_AnnotationPatternRegistry.register("conv")
def _annotate_conv(
    model: _torch.fx.GraphModule,
    quantization_config: _Optional[_AnnotationConfig],
    filter_fn: _Optional[_Callable[[_Node], bool]] = None,
) -> _Optional[_List[_List[_Node]]]:
    """
    Annotates input -> conv -> output
    """
    annotated_partitions = []
    for conv_dim in [1, 2, 3]:
        pattern_gm = _annotation_utils.get_conv_pattern(conv_dim=conv_dim, act_fn=None)
        annotated_partitions.extend(
            _annotation_utils.annotate_weighted_mod_pattern(
                model, pattern_gm, quantization_config, filter_fn
            )
        )

    return annotated_partitions


@_AnnotationPatternRegistry.register("linear_act")
def _annotate_linear_act(
    model: _torch.fx.GraphModule,
    quantization_config: _Optional[_AnnotationConfig],
    filter_fn: _Optional[_Callable[[_Node], bool]] = None,
) -> _Optional[_List[_List[_Node]]]:
    """
    Annotates input -> linear -> activation -> output
    """
    return _annotation_utils.annotate_linear_bn_act_helper(
        model, quantization_config, filter_fn, use_bn=False
    )


@_AnnotationPatternRegistry.register("linear_bn_act")
def _annotate_linear_bn_act(
    model: _torch.fx.GraphModule,
    quantization_config: _Optional[_AnnotationConfig],
    filter_fn: _Optional[_Callable[[_Node], bool]] = None,
) -> _Optional[_List[_List[_Node]]]:
    """
    Annotates input -> linear -> batch_norm -> activation -> output
    """
    return _annotation_utils.annotate_linear_bn_act_helper(
        model, quantization_config, filter_fn, use_bn=True
    )


@_AnnotationPatternRegistry.register("linear_bn")
def _annotate_linear_bn(
    model: _torch.fx.GraphModule,
    quantization_config: _Optional[_AnnotationConfig],
    filter_fn: _Optional[_Callable[[_Node], bool]] = None,
) -> _Optional[_List[_List[_Node]]]:
    """
    Annotates input -> linear -> batch_norm -> output
    """
    pattern_gm = _annotation_utils.get_linear_bn_pattern(
        act_fn=None, act_in_place=False
    )
    return _annotation_utils.annotate_weighted_mod_pattern(
        model, pattern_gm, quantization_config, filter_fn
    )


@_AnnotationPatternRegistry.register("linear")
def _annotate_linear(
    model: _torch.fx.GraphModule,
    quantization_config: _Optional[_AnnotationConfig],
    filter_fn: _Optional[_Callable[[_Node], bool]] = None,
) -> _Optional[_List[_List[_Node]]]:
    """
    Annotates input -> linear -> output
    """
    pattern_gm = _annotation_utils.get_linear_pattern(act_fn=None)
    return _annotation_utils.annotate_weighted_mod_pattern(
        model, pattern_gm, quantization_config, filter_fn
    )


@_AnnotationPatternRegistry.register("add_act")
def _annotate_add_act(
    model: _torch.fx.GraphModule,
    quantization_config: _Optional[_AnnotationConfig],
    filter_fn: _Optional[_Callable[[_Node], bool]] = None,
) -> _Optional[_List[_List[_Node]]]:
    """
    Annotates

    input_1 ---
               \
                --> add -> activation -> output
               /
    input_2 ---
    """
    ops = [_operator.add, _torch.add, _operator.iadd]
    return _annotation_utils.annotate_binary_op_helper(
        model, ops, quantization_config, filter_fn
    )


@_AnnotationPatternRegistry.register("add")
def _annotate_add(
    model: _torch.fx.GraphModule,
    quantization_config: _Optional[_AnnotationConfig],
    filter_fn: _Optional[_Callable[[_Node], bool]] = None,
) -> _Optional[_List[_List[_Node]]]:
    """
    Annotates

    input_1 ---
               \
                --> add -> output
               /
    input_2 ---
    """
    annotated_partitions = []
    ops = [_operator.add, _torch.add, _operator.iadd]
    for binary_op in ops:
        pattern_gm = _annotation_utils.get_binary_op_act_pattern(binary_op, None)
        annotated_partitions.extend(
            _annotation_utils.annotate_binary_op_act_pattern(
                model, pattern_gm, quantization_config, filter_fn
            )
        )
    return annotated_partitions


@_AnnotationPatternRegistry.register("mul_act")
def _annotate_mul_act(
    model: _torch.fx.GraphModule,
    quantization_config: _Optional[_AnnotationConfig],
    filter_fn: _Optional[_Callable[[_Node], bool]] = None,
) -> _Optional[_List[_List[_Node]]]:
    """
    Annotates

    input_1 ---
               \
                --> mul -> activation -> output
               /
    input_2 ---
    """
    ops = [_operator.mul, _torch.mul, _operator.imul]
    return _annotation_utils.annotate_binary_op_helper(
        model, ops, quantization_config, filter_fn
    )


@_AnnotationPatternRegistry.register("mul")
def _annotate_mul(
    model: _torch.fx.GraphModule,
    quantization_config: _Optional[_AnnotationConfig],
    filter_fn: _Optional[_Callable[[_Node], bool]] = None,
) -> _Optional[_List[_List[_Node]]]:
    """
    Annotates

    input_1 ---
               \
                --> mul -> output
               /
    input_2 ---
    """
    annotated_partitions = []
    ops = [_operator.mul, _torch.mul, _operator.imul]
    for binary_op in ops:
        pattern_gm = _annotation_utils.get_binary_op_act_pattern(binary_op, None)
        annotated_partitions.extend(
            _annotation_utils.annotate_binary_op_act_pattern(
                model, pattern_gm, quantization_config, filter_fn
            )
        )
    return annotated_partitions


@_AnnotationPatternRegistry.register("matmul_act")
def _annotate_matmul_act(
    model: _torch.fx.GraphModule,
    quantization_config: _Optional[_AnnotationConfig],
    filter_fn: _Optional[_Callable[[_Node], bool]] = None,
) -> _Optional[_List[_List[_Node]]]:
    """
    Annotates

    input_1 ---
               \
                --> matmul -> activation -> output
               /
    input_2 ---
    """
    return _annotation_utils.annotate_binary_op_helper(
        model, [_torch.matmul], quantization_config, filter_fn
    )


@_AnnotationPatternRegistry.register("matmul")
def _annotate_matmul(
    model: _torch.fx.GraphModule,
    quantization_config: _Optional[_AnnotationConfig],
    filter_fn: _Optional[_Callable[[_Node], bool]] = None,
) -> _Optional[_List[_List[_Node]]]:
    """
    Annotates

    input_1 ---
               \
                --> matmul -> output
               /
    input_2 ---
    """
    pattern_gm = _annotation_utils.get_binary_op_act_pattern(_torch.matmul, None)
    return _annotation_utils.annotate_binary_op_act_pattern(
        model, pattern_gm, quantization_config, filter_fn
    )


@_AnnotationPatternRegistry.register("max_pool1d")
def _annotate_max_pool1d(
    model: _torch.fx.GraphModule,
    quantization_config: _Optional[_AnnotationConfig],
    filter_fn: _Optional[_Callable[[_Node], bool]] = None,
) -> _Optional[_List[_List[_Node]]]:
    """
    Annotates

    input -> max_pool1d -> output
    """
    return _annotation_utils.annotate_unary_shared_observer_ops(
        model,
        [_torch.nn.MaxPool1d, _torch.nn.functional.max_pool1d, _torch.max_pool1d],
        quantization_config,
        filter_fn,
    )


@_AnnotationPatternRegistry.register("max_pool2d")
def _annotate_max_pool2d(
    model: _torch.fx.GraphModule,
    quantization_config: _Optional[_AnnotationConfig],
    filter_fn: _Optional[_Callable[[_Node], bool]] = None,
) -> _Optional[_List[_List[_Node]]]:
    """
    Annotates

    input -> max_pool2d ->  output
    """
    return _annotation_utils.annotate_unary_shared_observer_ops(
        model,
        [_torch.nn.MaxPool2d, _torch.nn.functional.max_pool2d, _torch.max_pool2d],
        quantization_config,
        filter_fn,
    )


@_AnnotationPatternRegistry.register("max_pool3d")
def _annotate_max_pool3d(
    model: _torch.fx.GraphModule,
    quantization_config: _Optional[_AnnotationConfig],
    filter_fn: _Optional[_Callable[[_Node], bool]] = None,
) -> _Optional[_List[_List[_Node]]]:
    """
    Annotates

    input -> max_pool3d -> output
    """
    return _annotation_utils.annotate_unary_shared_observer_ops(
        model,
        [_torch.nn.MaxPool3d, _torch.nn.functional.max_pool3d, _torch.max_pool3d],
        quantization_config,
        filter_fn,
    )


@_AnnotationPatternRegistry.register("adaptive_avg_pool1d")
def _annotate_adaptive_avg_pool1d(
    model: _torch.fx.GraphModule,
    quantization_config: _Optional[_AnnotationConfig],
    filter_fn: _Optional[_Callable[[_Node], bool]] = None,
) -> _Optional[_List[_List[_Node]]]:
    """
    Annotates

    input -> adaptive_avg_pool1d -> output
    """
    return _annotation_utils.annotate_unary_shared_observer_ops(
        model,
        [
            _torch.nn.AdaptiveAvgPool1d,
            _torch.nn.functional.adaptive_avg_pool1d,
            _torch.adaptive_avg_pool1d,
        ],
        quantization_config,
        filter_fn,
    )


@_AnnotationPatternRegistry.register("adaptive_avg_pool2d")
def _annotate_adaptive_avg_pool2d(
    model: _torch.fx.GraphModule,
    quantization_config: _Optional[_AnnotationConfig],
    filter_fn: _Optional[_Callable[[_Node], bool]] = None,
) -> _Optional[_List[_List[_Node]]]:
    """
    Annotates

    input -> adaptive_avg_pool2d -> output
    """
    return _annotation_utils.annotate_unary_shared_observer_ops(
        model,
        [_torch.nn.AdaptiveAvgPool2d, _torch.nn.functional.adaptive_avg_pool2d],
        quantization_config,
        filter_fn,
    )


@_AnnotationPatternRegistry.register("adaptive_avg_pool3d")
def _annotate_adaptive_avg_pool3d(
    model: _torch.fx.GraphModule,
    quantization_config: _Optional[_AnnotationConfig],
    filter_fn: _Optional[_Callable[[_Node], bool]] = None,
) -> _Optional[_List[_List[_Node]]]:
    """
    Annotates

    input -> adaptive_avg_pool3d -> output
    """
    return _annotation_utils.annotate_unary_shared_observer_ops(
        model,
        [_torch.nn.AdaptiveAvgPool3d, _torch.nn.functional.adaptive_avg_pool3d],
        quantization_config,
        filter_fn,
    )


@_AnnotationPatternRegistry.register("avg_pool1d")
def _annotate_avg_pool1d(
    model: _torch.fx.GraphModule,
    quantization_config: _Optional[_AnnotationConfig],
    filter_fn: _Optional[_Callable[[_Node], bool]] = None,
) -> _Optional[_List[_List[_Node]]]:
    """
    Annotates

    input -> avg_pool1d -> output
    """
    return _annotation_utils.annotate_unary_shared_observer_ops(
        model,
        [
            _torch.nn.AvgPool1d,
            _torch.nn.functional.avg_pool1d,
            _torch.avg_pool1d,
            _torch.mean,
        ],
        quantization_config,
        filter_fn,
    )


@_AnnotationPatternRegistry.register("avg_pool2d")
def _annotate_avg_pool2d(
    model: _torch.fx.GraphModule,
    quantization_config: _Optional[_AnnotationConfig],
    filter_fn: _Optional[_Callable[[_Node], bool]] = None,
) -> _Optional[_List[_List[_Node]]]:
    """
    Annotates

    input -> avg_pool2d -> output
    """
    return _annotation_utils.annotate_unary_shared_observer_ops(
        model,
        [
            _torch.nn.AvgPool2d,
            _torch.nn.functional.avg_pool2d,
            _torch._C._nn.avg_pool2d,
        ],
        quantization_config,
        filter_fn,
    )


@_AnnotationPatternRegistry.register("avg_pool3d")
def _annotate_avg_pool3d(
    model: _torch.fx.GraphModule,
    quantization_config: _Optional[_AnnotationConfig],
    filter_fn: _Optional[_Callable[[_Node], bool]] = None,
) -> _Optional[_List[_List[_Node]]]:
    """
    Annotates

    input -> avg_pool3d -> output
    """
    return _annotation_utils.annotate_unary_shared_observer_ops(
        model,
        [
            _torch.nn.AvgPool3d,
            _torch.nn.functional.avg_pool3d,
            _torch._C._nn.avg_pool3d,
        ],
        quantization_config,
        filter_fn,
    )


@_AnnotationPatternRegistry.register("flatten")
def _annotate_flatten(
    model: _torch.fx.GraphModule,
    quantization_config: _Optional[_AnnotationConfig],
    filter_fn: _Optional[_Callable[[_Node], bool]] = None,
) -> _Optional[_List[_List[_Node]]]:
    """
    Annotates

    input -> flatten -> output
    """
    return _annotation_utils.annotate_unary_shared_observer_ops(
        model,
        [
            _torch.nn.Flatten,
            _torch.flatten,
        ],
        quantization_config,
        filter_fn,
    )


class CoreMLQuantizer(_TorchQuantizer):
    """
    Annotates all recognized patterns using ``config``.

    Extends py:class:`torch.ao.quantization.quantizer.quantizer.Quantizer` to
    add support for quantization patterns supported by Core ML.

    Use it in conjunction with PyTorch 2.0 ``prepare_pt2e`` and ``prepare_qat_pt2e`` APIs
    for post training weight and activation quantization using calibration data and
    for quantization aware training (QAT).

    Example:

            .. code-block:: python

                import torch.nn as nn
                from torch.export import export_for_training
                from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_qat_pt2e

                from coremltools.optimize.torch.quantization._coreml_quantizer import CoreMLQuantizer

                model = nn.Sequential(
                    OrderedDict(
                        {
                            "conv": nn.Conv2d(1, 20, (3, 3)),
                            "relu1": nn.ReLU(),
                            "conv2": nn.Conv2d(20, 20, (3, 3)),
                            "relu2": nn.ReLU(),
                        }
                    )
                )

                loss_fn = define_loss()

                # initialize the annotator with quantization config
                config = LinearQuantizerConfig.from_dict(
                    {
                        "global_config": {
                            "quantization_scheme": "symmetric",
                        }
                    }
                )
                quantizer = CoreMLQuantizer(config)

                example_inputs = torch.randn(1, 1, 28, 28)

                # create export graph
                exported_model = export_for_training(model, (example_inputs,)).module()

                # prepare the model to insert FakeQuantize layers for QAT
                prepared_model = prepare_qat_pt2e(exported_model, quantizer)

                # use prepared model in your PyTorch training loop
                for inputs, labels in data:
                    output = prepared_model(inputs)
                    loss = loss_fn(output, labels)
                    loss.backward()
                    optimizer.step()
                    # turn observers/quantizers on/off depending on iteration number

                # convert operations to their quanitzed counterparts using parameters learnt via QAT
                model = convert_pt2e(prepared_model)
    """

    def __init__(self, config: _Optional[_LinearQuantizerConfig]):
        self._config = config

    def _annotate_all_patterns(
        self,
        model: _torch.fx.GraphModule,
        quantization_config: _Optional[_ModuleLinearQuantizerConfig],
        filter_fn: _Optional[_Callable[[_Node], bool]] = None,
    ):
        annotators = _AnnotationPatternRegistry.get_annotators()
        for _, annotator in annotators.items():
            annotation_config = _AnnotationConfig.from_quantization_config(
                quantization_config
            )
            annotator(model, annotation_config, filter_fn)

    def annotate(self, model: _torch.fx.GraphModule) -> _torch.fx.GraphModule:
        # First annotate all modules/operations which have name based configs
        module_name_list = list(self._config.module_name_configs.keys())
        for module_name, config in self._config.module_name_configs.items():
            self._annotate_all_patterns(
                model, config, _get_module_name_filter(module_name)
            )

        # Next annotate all modules/operations which have type based configs
        tp_list = list(self._config.module_type_configs.keys())
        for module_type, config in self._config.module_type_configs.items():
            self._annotate_all_patterns(
                model, config, _annotation_utils.get_object_type_filter(module_type)
            )

        # Annotate all other modules/operations
        self._annotate_all_patterns(
            model,
            self._config.global_config,
            _annotation_utils.get_not_object_type_or_name_filter(
                tp_list, module_name_list
            ),
        )
        return model

    def validate(self, model: _torch.fx.GraphModule) -> None:
        pass
