# Copyright (c) 2024, Apple Inc. All rights reserved.
#
# Use of this source code is governed by a BSD-3-clause license that can be
# found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from collections import defaultdict
from typing import Dict, List, Optional, Union

import numpy as np
from tqdm import tqdm

from coremltools import _SPECIFICATION_VERSION_IOS_17
from coremltools import _logger as logger
from coremltools.converters.mil.converter import mil_convert as _mil_convert
from coremltools.converters.mil.frontend.milproto import load as _milproto_to_pymil
from coremltools.converters.mil.mil.passes.graph_pass import PassOption
from coremltools.converters.mil.mil.passes.pass_registry import PASS_REGISTRY
from coremltools.models import MLModel as _MLModel
from coremltools.models import utils as _model_utils
from coremltools.optimize.coreml import OptimizationConfig as _OptimizationConfig

from ._model_debugger import ModelDebugger
from ._quantization_passes import (
    insert_prefix_quantize_dequantize_pair as _insert_prefix_quantize_dequantize_pair,
)


def linear_quantize_activations(
    mlmodel: _MLModel,
    config: _OptimizationConfig,
    sample_data: List[Dict[Optional[str], np.ndarray]],
    calibration_op_group_size: int = -1,
):
    """
    Utility function to convert a float precision MLModel of type ``mlprogram``, which uses
    float-precision activations, into a compressed MLModel that uses n-bit activations. Currently, only n=8
    is suppported.

    This is achieved by feeding real sample data into the input MLModel, calibrating the resulting float activation values,
    converting the calibrated values into ``quantize`` and ``dequantize`` op pairs, and inserting those
    op pairs into the new MLModel instance where activations get quantized.

    Use this function with ``linear_quantize_weights`` for 8-bit activation and 8-bit weight linear quantization.
    It's also compatible for use with other weight compression methods.

    Parameters
    ----------
    mlmodel: MLModel
        Model to be quantized. This MLModel should be of type ``mlprogram``.

    config: OptimizationConfig
        An :py:class:`OptimizationConfig` object that specifies the parameters for activation quantization.

    sample_data: List
        Data used to characterize statistics of the activation values of the original float precision model.
        Expects a list of sample input dictionaries, which should have the same format as the data used in `.predict`
        method for the mlmodel. More specifically, the input name need to be specified in the data, unless it's a single
        input model where the name will be auto inferred.

    calibration_op_group_size: int
        While running inference during calibration, only have `calibration_op_group_size` of intermediate outputs
        appended to outputs at a time. If the model is very large, it could lead to the temperary model having
        thousands of outputs, which may lead to model hanging forever during model loading. To work around this
        issue, intermediate outputs are grouped into smaller groups, where each time a temperary model will only
        have `calibration_op_group_size` outputs. By default (op_group_size = -1), op_group_size is equal to the
        number of valid intermediate ops.

    Returns
    -------
    model: MLModel
        The activation quantized MLModel instance.

    Examples
    --------
    .. sourcecode:: python

        import coremltools as ct
        import coremltools.optimize as cto

        model = ct.coreml.models.MLModel("my_model.mlpackage")
        activation_config = cto.coreml.OptimizationConfig(
            global_config=cto.coreml.experimental.OpActivationLinearQuantizerConfig(
                mode="linear_symmetric"
            )
        )
        compressed_model_a8 = cto.coreml.experimental.linear_quantize_activations(
            model, activation_config, sample_data
        )

        # (Optional) It's recommended to use with linear_quantize_weights.
        weight_config = cto.coreml.OptimizationConfig(
            global_config=cto.OpLinearQuantizerConfig(mode="linear_symmetric")
        )
        compressed_model_w8a8 = cto.linear_quantize_weights(compressed_model_a8, weight_config)
    """
    # Validate Sample data. If the sample data name is not provided, try to infer it.
    for sample in sample_data:
        if None in sample.keys():
            input_spec = mlmodel.get_spec().description.input
            if len(sample.keys()) > 1 or len(input_spec) > 1:
                raise ValueError(
                    "When the model has multiple inputs, please provide the name for each data in `sample_data`"
                )
            inferred_input_name = input_spec[0].name
            sample[inferred_input_name] = sample[None]
            del sample[None]

    ### Apply four major graph passes in order.

    # Insert prefix quantize/dequantize pairs to valid patterns.
    logger.info("Running compression pass linear_quantize_activations phase 1/3 ...")
    linear_activation_quantizer = PASS_REGISTRY[
        "compression::insert_prefix_quantize_dequantize_pair"
    ]
    linear_activation_quantizer = _insert_prefix_quantize_dequantize_pair(
        config, fake_compression=False
    )
    linear_activation_quantizer.set_options([PassOption("config", config)])
    activation_stats = _get_activation_calibration_stats(mlmodel, sample_data)
    linear_activation_quantizer.set_options([PassOption("activation_stats", activation_stats)])

    prog = _model_utils._apply_graph_pass(
        mlmodel,
        linear_activation_quantizer,
        spec_version=_SPECIFICATION_VERSION_IOS_17,
        pymil_load_func=_milproto_to_pymil.load,
        skip_model_load=True,  # Save memony
        return_pymil_prog=True,
    )

    # Insert suffix quantize/dequantize pairs to valid patterns.
    logger.info("Running compression pass linear_quantize_activations phase 2/3 ...")
    graph_pass = PASS_REGISTRY["compression::insert_suffix_quantize_dequantize_pair"]
    graph_pass.set_options([PassOption("config", config)])
    graph_pass.set_options([PassOption("activation_stats", activation_stats)])
    graph_pass(prog)
    prog.validate()

    # Re-use exsiting path to dedup quantize/dequantize operations.
    logger.info("Running compression pass linear_quantize_activations phase 3/3 ...")
    graph_pass = PASS_REGISTRY["common::dequantize_quantize_pair_elimination"]
    graph_pass(prog)
    prog.validate()

    # Convert the pymil program (prog) back to mlmodel
    model_spec = mlmodel.get_spec()
    specification_version = max(model_spec.specificationVersion, _SPECIFICATION_VERSION_IOS_17)
    mlmodel_activation_quantized = _mil_convert(
        prog,
        convert_to="mlprogram",
        convert_from="milinternal",
        specification_version=specification_version,
        compute_units=mlmodel.compute_unit,
        model_description=model_spec.description,
        skip_model_load=False,  # Must be False to avoid manually re-loading from disk before running prediction.
    )
    return mlmodel_activation_quantized


def _update_tensor_range(
    tensor_name: str,
    tensor_value: Union[int, float],
    activation_stats_dict: Dict[str, Dict[str, float]],
) -> None:
    tensor_min = np.min(np.array(tensor_value).flatten())
    tensor_max = np.max(np.array(tensor_value).flatten())
    activation_stats_dict[tensor_name]["rmin"] = tensor_min
    activation_stats_dict[tensor_name]["rmax"] = tensor_max
    if tensor_name in activation_stats_dict:
        activation_stats_dict[tensor_name]["rmin"] = min(
            tensor_min, activation_stats_dict[tensor_name]["rmin"]
        )
        activation_stats_dict[tensor_name]["rmax"] = max(
            tensor_max, activation_stats_dict[tensor_name]["rmax"]
        )
    else:
        activation_stats_dict[tensor_name]["rmin"] = tensor_min
        activation_stats_dict[tensor_name]["rmax"] = tensor_max


def _combine_lists_with_common_elements(data: List[List[str]]) -> List[List[str]]:
    """
    Parameters
    ----------
    data: list[list[]]
        data is a list of lists with strings.

    Returns
    -------
    merged: combined lists with common elements.

    Example
    -------
    input: [["conv0", "conv1", "conv2"], ["conv0", "conv3"], ["relu0"]]
    output: [["conv0", "conv1", "conv2", "conv3"], ["relu0"]]
    """

    merged = []
    for item in data:
        item_set = set(item)
        not_exsit = True
        for result in merged:
            if result & item_set:
                result.update(item_set)
                not_exsit = False
                break
        if not_exsit:
            merged.append(item_set)
    return merged


def _adjust_concat_surrounding_activation_stats(
    concat_op_info_list: List[List[str]], activation_stats_dict: Dict[str, Dict[str, float]]
) -> None:
    """
    Adjust the activation calibration stats of inputs/outputs to the same concat ops to maximize hardware efficiency.
    Tensor values of inputs/outputs to the same concat op should share same range (same min/max), so the quantized
    concat could be surrounded by quantize/dequantize pairs with same scale and zero point values.

    Example
    -------
    - concat 1 -
    inputs:  "input_1", "input_2", "input_3"
    output:  "output_1"

    - concat 2 -
    inputs:  "input_1", "input_4"
    output:  "output_2"

    Input/output tensors range of concat 1 should be identical.
    Input/output tensors range of concat 2 should be identical.
    "input_1" is in both, which means activation calibration stats of all 6 tensors above should be identical.
    """

    if concat_op_info_list is None:
        return

    # Merge tensor names which should have identical values, to the same list.
    concat_list_adjusted = _combine_lists_with_common_elements(concat_op_info_list)

    for concat_group in concat_list_adjusted:
        group_rmin_list, group_rmax_list = [], []

        for tensor_name in concat_group:
            # Some tensor_name may not have rmin/rmax if the calibration failed before.
            if tensor_name in activation_stats_dict:
                group_rmin_list.append(activation_stats_dict[tensor_name]["rmin"])
                group_rmax_list.append(activation_stats_dict[tensor_name]["rmax"])

        if len(group_rmin_list) == 0:
            raise ValueError(
                "None of the calibration run succeeded. Please check logs about calibrating sample failures."
            )
        group_rmin, group_rmax = min(group_rmin_list), max(group_rmax_list)

        for tensor_name in concat_group:
            if tensor_name in activation_stats_dict:
                activation_stats_dict[tensor_name]["rmin"] = group_rmin
                activation_stats_dict[tensor_name]["rmax"] = group_rmax


def _get_activation_calibration_stats(
    fpmodel: _MLModel,
    sample_data: List[Dict[str, np.ndarray]],
    calibration_op_group_size: int = -1,
) -> Dict[str, Dict[str, float]]:
    """
    Calibration and store a dict of intermediate tensor stats.
    E.g. activation_stats_dict = {tensor_0: {rmin: 0.2, rmax: 3.8}, tensor_1: {rmin: 4.5, rmax: 12.6}}}
    Parameters
    ----------
    fpmodel: MLModel
        Path to fp16/fp32 "model.mlpackage". (Expect the orginal mlmodel, not the one with quantize and dequant op)
    sample_data: list[dict]
        Data for calibration.
    calibration_op_group_size: int
        While running inference during calibration, only have `calibration_op_group_size` of intermediate outputs
        appended to outputs at a time. If the model is very large, it could lead to the temperary model having
        thousands of outputs, which may lead to model hanging forever during model loading. To work around this
        issue, intermediate outputs are grouped into smaller groups, where each time a temperary model will only
        have `calibration_op_group_size` outputs. By default (op_group_size = -1), op_group_size is equal to the
        number of valid intermediate ops.

    Returns
    -------
    activation_calibration_stats: dict
    """
    logger.warning(
        "Running compression pass linear_quantize_activations: start calibrating {} samples".format(
            len(sample_data)
        )
    )
    logger.warning(
        "Running compression pass linear_quantize_activations: calibration may take a while ..."
    )

    debugger = ModelDebugger(fpmodel)
    activation_stats_dict = defaultdict(dict)
    intermediate_output_names = debugger.get_intermediate_output_names(
        lambda op: (op.spec.type != "const")
    )

    # Get data ranges for all inputs.
    for data in sample_data:
        for input_name in data:
            _update_tensor_range(input_name, data[input_name], activation_stats_dict)

    # The last few elements in intermediate_output_names might be output.
    # We don't maintain min/max value for an output tensor.
    # If it's an output tensor we exclude it, otherwise include it.
    model_spec = fpmodel.get_spec()
    output_count = len(fpmodel.get_spec().description.output)
    output_names = []
    for i in range(0, output_count):
        output_name = model_spec.description.output[i].name
        output_names.append(output_name)

    for intermediate_output_name in intermediate_output_names:
        if intermediate_output_name in output_names:
            intermediate_output_names.remove(intermediate_output_name)

    # Get data ranges for all intermeditate outputs.
    for data in tqdm(
        sample_data,
        desc="Running compression pass linear_quantize_activations",
        unit=" calibration samples",
    ):
        debugger.step(
            inputs=data,
            activation_stats_dict=activation_stats_dict,
            intermediate_output_names=intermediate_output_names,
            op_group_size=calibration_op_group_size,
        )

    # Handle a special case - concat ops.
    _adjust_concat_surrounding_activation_stats(
        debugger._get_concat_op_info(), activation_stats_dict
    )

    return activation_stats_dict
