#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import numpy as _np
import logging as _logging
from coremltools.models import neural_network as neural_network
from coremltools.proto import NeuralNetwork_pb2
from coremltools.converters.mil.mil.types.symbolic import (
    is_variadic,
    any_symbolic,
    is_symbolic,
)
from coremltools.converters.mil.mil.types import np_dtype_to_py_type
from coremltools.converters.mil.mil import types
from coremltools.converters.mil.mil.ops.registry import SSAOpRegistry
from coremltools.models.neural_network.quantization_utils import (
    _convert_array_to_nbit_quantized_bytes,
)
from tqdm import tqdm as _tqdm
from .mil_to_nn_mapping_registry import *


def convert_ops(const_context, builder, ops, outputs):
    """
    const_context: list[set of str]: const name for v1 & v2 (the same)
    builder: neural_network.NeuralNetworkBuilder
    ops: list[Operation], usually from Block.operations.
    outputs: list[Var]. block outputs
    """

    const_context.append(set())
    custom_ops = SSAOpRegistry.custom_ops
    for op in _tqdm(ops, desc="Translating MIL ==> NeuralNetwork Ops", unit=" ops"):
        if op.op_type in custom_ops:
            mapper = MIL_TO_NN_MAPPING_REGISTRY["custom_op"]
        elif op.op_type in MIL_TO_NN_MAPPING_REGISTRY:
            mapper = MIL_TO_NN_MAPPING_REGISTRY[op.op_type]
        else:
            msg = ("Op {} is used in the source model. This op is not supported "
                   "by the NeuralNetwork (compatibility with MacOS < 12, iOS < 15) model "
                   "type. To successfully convert this model, convert to the ML Program "
                   "model type (minimum target MacOS 12, iOS 15 and later).\n"
                   "Use coremltools.convert(..., convert_to=\"mlprogram\") to convert to ML Program.\n"
                   "block: {}")
            raise NotImplementedError(msg.format(op.op_type, op.enclosing_block))
        # const is globally shared in nn.
        mapper(const_context, builder, op)

    for ov in outputs:
        # If block return value is a const, we need to add it.
        if ov.op is None:
            continue  # placeholder
        if ov.op.op_type == "const":
            add_const(const_context, builder, ov.name, ov.val)
    const_context.pop()


def make_input(const_context, builder, variables):
    """
    Ensure that variables, if const, are added to builder.

    variables: list[Var] or Var or str. Inputs for an nn layer.

    Returns:
        list[str] or str: variables' names.
    """
    if isinstance(variables, (list, tuple)):
        return [make_input(const_context, builder, v) for v in variables]
    if isinstance(variables, str):
        return variables

    v = variables  # variables is Var
    if v.op is not None and v.op.op_type == "const" and v.name not in const_context[-1]:
        add_const(const_context, builder, v.name, v.val)
    return v.name

def to_py_type(val):
    """Convert numpy val to python primitive equivalent. Ex:

    Given: val = np.array([True, False])
    Returns: [True, False]

    Given: val = np.array(32, dtype=np.int)
    Returns 32
    """
    if not isinstance(val, (_np.ndarray, _np.generic)):
        return val

    # val is np.ndarray or np.generic
    is_np_scalar = isinstance(val, _np.generic) or val.shape == ()
    py_type = np_dtype_to_py_type(val.dtype)
    if is_np_scalar:
        return py_type(val)
    # flatten them to 1D array
    val = val.flatten()
    return tuple(py_type(v) for v in val)

def _convert_pool(const_context, builder, op, mode, exclude_padding_from_average=True):
    num_spatial_dimensions = len(op.kernel_sizes.val)
    op_pad = op.pad.val if op.pad_type.val == 'custom' \
        else [0] * num_spatial_dimensions * 2
    if num_spatial_dimensions == 1:
        builder.add_expand_dims(
            name=op.name + "_expanded",
            input_name=op.x.name,
            output_name=op.name + "_expanded",
            axes=[-2],
        )
        padding_type = op.pad_type.val.upper()
        # nn's add_pool function does not support CUSTOM padding,
        # but VALID padding supports user-defined padding amounts.
        # Therefore we map CUSTOM padding to VALID padding.
        padding_type = "VALID" if padding_type == "CUSTOM" else padding_type
        builder.add_pooling(
            name=op.name,
            height=1,
            width=op.kernel_sizes.val[-1],
            stride_height=1,
            stride_width=op.strides.val[-1],
            layer_type=mode.upper(),
            padding_type="INCLUDE_LAST_PIXEL" if op.ceil_mode.val else padding_type,
            input_name=make_input(const_context, builder, op.name + "_expanded"),
            output_name=op.name + "_pool",
            exclude_pad_area=exclude_padding_from_average,
            padding_top=0,
            padding_bottom=0,
            padding_left=op_pad[0],
            padding_right=op_pad[1],
            is_global=False,
        )
        builder.add_squeeze(
            name=op.name + "_squeeze",
            input_name=op.name + "_pool",
            output_name=op.outputs[0].name,
            axes=[-2],
        )
    elif num_spatial_dimensions == 2:
        padding_type = op.pad_type.val.upper()
        # nn's add_pool function does not support CUSTOM padding,
        # but VALID padding supports user-defined padding amounts.
        # Therefore we map CUSTOM padding to VALID padding.
        padding_type = "VALID" if padding_type == "CUSTOM" else padding_type
        builder.add_pooling(
            name=op.name,
            height=op.kernel_sizes.val[-2],
            width=op.kernel_sizes.val[-1],
            stride_height=op.strides.val[-2],
            stride_width=op.strides.val[-1],
            layer_type=mode.upper(),
            padding_type="INCLUDE_LAST_PIXEL" if op.ceil_mode.val else padding_type,
            input_name=make_input(const_context, builder, op.x),
            output_name=op.outputs[0].name,
            exclude_pad_area=exclude_padding_from_average,
            padding_top=op_pad[0],
            padding_bottom=op_pad[1],
            padding_left=op_pad[2],
            padding_right=op_pad[3],
            is_global=False,
        )
    elif num_spatial_dimensions == 3:
        builder.add_pooling3d(
            name=op.name,
            input_name=make_input(const_context, builder, op.x),
            output_name=op.outputs[0].name,
            pooling_type=mode.upper(),
            kernel_depth=op.kernel_sizes.val[-3],
            kernel_height=op.kernel_sizes.val[-2],
            kernel_width=op.kernel_sizes.val[-1],
            stride_depth=op.strides.val[-3],
            stride_height=op.strides.val[-2],
            stride_width=op.strides.val[-1],
            padding_mode=op.pad_type.val,
            custom_padding_front=op_pad[0],
            custom_padding_back=op_pad[1],
            custom_padding_top=op_pad[2],
            custom_padding_bottom=op_pad[3],
            custom_padding_left=op_pad[4],
            custom_padding_right=op_pad[5],
            average_pooling_count_excludes_padding=exclude_padding_from_average,
        )
    else:
        raise ValueError(
            "Unsupported number of spatial dimensions. Maximum is 3, but got %s"
            % num_spatial_dimensions
        )


def _try_convert_global_pool(const_context, builder, op, mode):
    """
    Optional performance optimization pass that tries to lower spatial
    reduce_mean / reduce_max to global_avg_pool / global_max_pool.
    Return True if the lowering happened, otherwise return False to
    continue as normal reduction op.
    """
    rank = op.x.rank
    if is_variadic(rank) or rank not in {4, 5}:
        return False
    keep_dims = op.keep_dims.val
    if keep_dims is False:
        return False

    if op.axes is not None:
        axes = op.axes.val
        axes = sorted([rank + axis if axis < 0 else axis for axis in axes])

        if tuple(op.outputs[0].shape[:-2]) != tuple(op.inputs["x"].shape[:-2]):
            return False
        if not all([s == 1 for s in op.outputs[0].shape[-2:]]):
            return False
    builder.add_pooling(
        name=op.name,
        height=0,
        width=0,
        stride_height=0,
        stride_width=0,
        layer_type=mode.upper(),
        padding_type="valid".upper(),
        input_name=make_input(const_context, builder, op.x),
        output_name=op.outputs[0].name,
        is_global=True,
    )
    return True


def add_const(const_context, builder, name, val):
    """
    const_context (list of set of str): const names added to v1 builder. Const names are
    identical between v2 and v1

    name (str): name of const. Should be the same for v1 and v2.
    val: np.ndarray

    No return values as `name` is the name of const in v1.

    Comment: we don't need to add scalar const as they are just fields in
             layer proto message in NN.
             If we really need a const scalar, we upcast it to rank-1.

    """
    for const_set in const_context:
        if name in const_set:
            _logging.warning("Const {} was already added.".format(name))
            return
    if not isinstance(val, (_np.ndarray, _np.generic)):
        val = _np.array([val])
    if val.dtype != _np.float:
        # nn proto only supports float32 activation. (e.g., pred in cond op
        # needs to be converted to float)
        val = val.astype(_np.float)
    rank = len(val.shape)
    if rank == 0:
        builder.add_load_constant_nd(
            name=name, output_name=name, constant_value=val.reshape([1]), shape=[1]
        )
    else:
        builder.add_load_constant_nd(
            name=name, output_name=name, constant_value=val, shape=val.shape
        )
    const_context[-1].add(name)
    _logging.info("added const {} for builder {}".format(name, builder))


# Helper routines for recurrent layers
def _expand_dim(builder, node_name, input_name, axes):
    builder.add_expand_dims(
        name=node_name, input_name=input_name, output_name=node_name, axes=axes
    )


def _squeeze(builder, node_name, input_name, axes):
    builder.add_squeeze(
        name=node_name, input_name=input_name, output_name=node_name, axes=axes
    )


def _split(x, sections, axis=0):
    if x is None:
        return None
    if x.shape[axis] % sections != 0:
        raise ValueError(
            "Cannot split axis {} into {} sections for input of shape {}".format(
                axis, sections, x.shape
            )
        )
    return _np.split(x, sections, axis=axis)


@register_mil_to_nn_mapping
def avg_pool(const_context, builder, op):
    _convert_pool(
        const_context=const_context,
        builder=builder,
        op=op,
        mode="average",
        exclude_padding_from_average=op.exclude_padding_from_average.val,
    )


@register_mil_to_nn_mapping
def band_part(const_context, builder, op):
    builder.add_matrix_band_part(
        name=op.name,
        input_name=make_input(const_context, builder, op.x),
        output_name=op.outputs[0].name,
        num_lower=op.lower.val,
        num_upper=op.upper.val,
    )


@register_mil_to_nn_mapping
def batch_norm(const_context, builder, op):
    channels = op.x.shape[1]
    gamma = _np.array([1.0] * channels) if op.gamma is None else op.gamma.val
    beta = _np.array([0.0] * channels) if op.beta is None else op.beta.val

    x_name = make_input(const_context, builder, op.x)
    out_name = op.outputs[0].name

    if op.x.rank == 3:
        x_name = op.name + "_expanded"
        builder.add_expand_dims(
            name=x_name, input_name=op.x.name, output_name=x_name, axes=[-2],
        )
        out_name += "_batch_norm"

    builder.add_batchnorm(
        name=op.name,
        channels=channels,
        gamma=gamma,
        beta=beta,
        mean=op.mean.val,
        variance=op.variance.val,
        input_name=x_name,
        output_name=out_name,
        compute_mean_var=False,
        instance_normalization=False,
        epsilon=op.epsilon.val,
    )

    # Squeeze added `Width` dimension for 1d case
    if op.x.rank == 3:
        x_name = op.name + "_squeeze"
        builder.add_squeeze(
            name=x_name,
            input_name=out_name,
            output_name=op.outputs[0].name,
            axes=[-2],
        )

@register_mil_to_nn_mapping
def const(const_context, builder, op):
    # const in V2 are added to V1 lazily.
    pass


def conv_helper(const_context, builder, op):
    # v2 x: (n, C_in/groups, spatial_dims)
    x_name = make_input(const_context, builder, op.x)
    out_name = op.outputs[0].name

    is_conv1d = op.x.rank == 3
    is_conv2d = op.x.rank == 4
    is_conv3d = op.x.rank == 5
    if not (is_conv1d or is_conv2d or is_conv3d):
        raise ValueError(
            "Input tensor rank '{}' is not one of '{}'.".format(op.x.rank, (3, 4, 5),)
        )
    if is_conv1d:
        x_name = op.name + "_expand_dim"
        out_name += "_expanded"
        builder.add_expand_dims(
            name=x_name, input_name=op.x.name, output_name=x_name, axes=[-2],
        )
    # `x_name` is guaranteed to be (n, C_in/groups, spatial_dims) for 1D and 2D convolution
    # W_v1 wil be np.ndarray (if W is const at compile time) or None
    # (if W is not known at compile time).
    weights = None
    input_names = [x_name]
    if op.weight.val is not None:
        # v2 convolution (conv3d) expects weights to have shape (C_out, C_in/groups, spatial_dims)
        # v1 convolution expects (H, W, C_in/groups, C_out) or (D, H, W, C_in/groups, C_out)
        weights = op.weight.val
        if is_conv1d:
            weights = _np.expand_dims(op.weight.val, -2)
        if is_conv1d or is_conv2d:
            weights = _np.transpose(weights, [2, 3, 1, 0])
    else:
        # op.weight is not const at compile time.
        # When weight is dynamic, v1 convolution expects weight to be
        # (C_out, C_in/groups, H, W)
        # TODO 3D convolution doesn't support dynamic weights:
        if is_conv3d:
            raise ValueError("3D Convolution doesn't support dynamic weights.")
        weights_name = op.weight.name
        if is_conv1d:
            weights_name += "_expand_dim"
            builder.add_expand_dims(
                name=weights_name,
                input_name=op.weight.name,
                output_name=weights_name,
                axes=[-2],
            )
        input_names.append(weights_name)

    # padding
    padding_mode = op.pad_type.val
    pad = {}
    if padding_mode == "custom":
        if is_conv1d:
            padding_mode = "valid"
            pad["padding_top"] = 0
            pad["padding_bottom"] = 0
            pad["padding_left"] = op.pad.val[0]
            pad["padding_right"] = op.pad.val[1]
        elif is_conv2d:
            padding_mode = "valid"
            pad["padding_top"] = op.pad.val[0]
            pad["padding_bottom"] = op.pad.val[1]
            pad["padding_left"] = op.pad.val[2]
            pad["padding_right"] = op.pad.val[3]
        else:
            pad["padding_front"] = op.pad.val[0]
            pad["padding_back"] = op.pad.val[1]
            pad["padding_top"] = op.pad.val[2]
            pad["padding_bottom"] = op.pad.val[3]
            pad["padding_left"] = op.pad.val[4]
            pad["padding_right"] = op.pad.val[5]

    has_bias = op.bias is not None
    groups = op.groups.val

    rank_factor = 1
    if is_conv2d:
        rank_factor = 2
    elif is_conv3d:
        rank_factor = 3

    strides = op.strides.val.tolist()
    dilations = op.dilations.val.tolist()
    if is_conv1d:
        dilations = dilations[:-1] + [1] + dilations[-1:]
        strides = strides[:-1] + [1] + strides[-1:]

    if weights is not None and op.op_type == "conv_quantized":
        nbits = op.nbits.val
        weights = _convert_array_to_nbit_quantized_bytes(weights.flatten(), nbits).tobytes()
        quantization_type = op.quantization_type.val
        quant_bias = op.quant_bias.val
        quant_scale = op.quant_scale.val
    else:
        quantization_type = None
        nbits = None
        quant_bias = None
        quant_scale = None

    if is_conv1d or is_conv2d:
        builder.add_convolution(
            name=out_name,
            kernel_channels=op.weight.shape[1],
            output_channels=op.weight.shape[0],
            height= 1 if is_conv1d else op.weight.shape[2],
            width= op.weight.shape[2] if is_conv1d else op.weight.shape[3],
            stride_height=strides[0],
            stride_width=strides[1],
            border_mode=padding_mode,
            groups=groups,
            W=weights,
            b=op.bias.val if has_bias else None,
            has_bias=has_bias,
            is_deconv=False,
            input_name=input_names,
            output_name=out_name,
            dilation_factors=dilations,
            quantization_type=quantization_type,
            nbits=nbits,
            quant_bias=quant_bias,
            quant_scale=quant_scale,
            **pad  # Python 2.7.16 will fail with a syntax error if a comma is included after `**pad`
        )

        # Squeeze added `Width` dimension for 1d case
        if is_conv1d:
            x_name = op.name + "expand_dim"
            builder.add_squeeze(
                name=op.name,
                input_name=out_name,
                output_name=op.outputs[0].name,
                axes=[-2],
            )

    if is_conv3d:
        builder.add_convolution3d(
            name=op.name,
            input_channels=op.weight.shape[1] * groups,
            output_channels=op.weight.shape[0],
            depth=op.weight.shape[2],
            height=op.weight.shape[3],
            width=op.weight.shape[4],
            W=op.weight.val,
            b=op.bias.val if has_bias else None,
            has_bias=has_bias,
            groups=groups,
            stride_depth=strides[0],
            stride_height=strides[1],
            stride_width=strides[2],
            dilation_depth=dilations[0],
            dilation_height=dilations[1],
            dilation_width=dilations[2],
            padding_mode=padding_mode,
            is_deconv=False,
            output_shape=None,
            input_name=input_names,
            output_name=out_name,
            **pad  # Python 2.7.16 will fail with a syntax error if a comma is included after `**pad`
        )

@register_mil_to_nn_mapping
def conv(const_context, builder, op):
    conv_helper(const_context, builder, op)


@register_mil_to_nn_mapping()
def conv_quantized(const_context, builder, op):
    conv_helper(const_context, builder, op)


@register_mil_to_nn_mapping
def cumsum(const_context, builder, op):
    input_names = make_input(const_context, builder, [op.x])
    builder.add_cumsum(
        name=op.name,
        input_names=input_names,
        output_name=op.outputs[0].name,
        axis=op.axis.val,
        reverse=op.reverse.val,
        exclusive=op.exclusive.val,
    )


def _add_elementwise_unary(
    const_context, builder, op, mode, output_name=None, **kwargs
):
    output_name = output_name if output_name else op.outputs[0].name
    name = output_name if output_name else op.name
    if mode in ["sqrt", "rsqrt", "inverse", "power", "exp", "log", "abs", "threshold"]:
        builder.add_unary(
            name=name,
            input_name=make_input(const_context, builder, op.x),
            output_name=output_name,
            mode=mode,
            **kwargs
        )
    else:
        add_func = getattr(builder, "add_" + mode, None)
        if add_func is None:
            _logging.error(
                "Elementwise unary method {} not found in builder.".format(mode)
            )
        add_func(
            name=name,
            input_name=make_input(const_context, builder, op.x),
            output_name=output_name,
            **kwargs
        )


def _add_elementwise_binary(
    const_context, builder, op, mode, output_name=None, **kwargs
):
    output_name = output_name if output_name else op.outputs[0].name
    name = output_name if output_name else op.name
    if mode in ["add", "multiply"]:
        params = {"name": name, "output_name": output_name, "mode": mode.upper()}
        if op.x.val is not None and op.x.rank == 0 and _np.isfinite(op.x.val):
            params["input_names"] = make_input(const_context, builder, [op.y])
            val = op.x.val if not isinstance(op.x.val, _np.float16) else op.x.val.astype(_np.float32)
            params["alpha"] = to_py_type(val)
            builder.add_elementwise(**params)
            return
        elif op.y.val is not None and op.y.rank == 0 and _np.isfinite(op.y.val):
            params["input_names"] = make_input(const_context, builder, [op.x])
            val = op.y.val if not isinstance(op.y.val, _np.float16) else op.y.val.astype(_np.float32)
            params["alpha"] = to_py_type(val)
            builder.add_elementwise(**params)
            return
    elif mode in ["equal", "not_equal"]:
        add_func = getattr(builder, "add_" + mode, None)
        params = {"name": name, "output_name": output_name}
        if op.x.val is not None and op.x.rank == 0 and _np.isfinite(op.x.val):
            params["input_names"] = make_input(const_context, builder, [op.y])
            val = op.x.val if not isinstance(op.x.val, _np.float16) else op.x.val.astype(_np.float32)
            params["alpha"] = to_py_type(val)
            add_func(**params)
            return
        elif op.y.val is not None and op.y.rank == 0 and _np.isfinite(op.y.val):
            params["input_names"] = make_input(const_context, builder, [op.x])
            val = op.y.val if not isinstance(op.y.val, _np.float16) else op.y.val.astype(_np.float32)
            params["alpha"] = to_py_type(val)
            add_func(**params)
            return
    elif mode in ["greater_than", "greater_equal", "less_than", "less_equal"]:
        params = {"name": name, "output_name": output_name}
        if op.x.val is not None and op.x.rank == 0 and _np.isfinite(op.x.val):
            params["input_names"] = make_input(const_context, builder, [op.y])
            val = op.x.val if not isinstance(op.x.val, _np.float16) else op.x.val.astype(_np.float32)
            params["alpha"] = to_py_type(val)
            if "less" in mode:
                params["use_greater_than_equal"] = mode.endswith("_equal")
                builder.add_greater_than(**params)
            elif "greater" in mode:
                params["use_less_than_equal"] = mode.endswith("_equal")
                builder.add_less_than(**params)
            return
        elif op.y.val is not None and op.y.rank == 0 and _np.isfinite(op.y.val):
            params["input_names"] = make_input(const_context, builder, [op.x])
            val = op.y.val if not isinstance(op.y.val, _np.float16) else op.y.val.astype(_np.float32)
            params["alpha"] = to_py_type(val)
            if "greater" in mode:
                params["use_greater_than_equal"] = mode.endswith("_equal")
                builder.add_greater_than(**params)
            elif "less" in mode:
                params["use_less_than_equal"] = mode.endswith("_equal")
                builder.add_less_than(**params)
            return

    if op.x.val is not None:
        add_const(const_context, builder, op.x.name, op.x.val)
    if op.y.val is not None:
        if mode == "pow":
            _add_elementwise_unary(
                const_context,
                builder,
                op,
                "power",
                output_name=output_name,
                alpha=op.y.val,
            )
            return
        add_const(const_context, builder, op.y.name, op.y.val)


    if mode in {"add", "multiply", "max", "min"} and op.x.shape == op.y.shape:
        builder.add_elementwise(
            name=name,
            input_names=make_input(const_context, builder, [op.x, op.y]),
            output_name=output_name,
            mode=mode.upper(),
        )
        return

    # the broadcast feature in the elementwise layer is hardcoded to 4D or less
    # for the 5d tensor, we need to use broadcasable layers instead.
    if mode in {"add", "multiply", "subtract"} and op.x.rank < 5 and op.y.rank < 5:
        shape_x = _np.array([1] * (5 - op.x.rank) + list(op.x.shape))
        shape_y = _np.array([1] * (5 - op.y.rank) + list(op.y.shape))

        internal_x = internal_y = None
        if all(shape_x == 1):
            internal_y = op.x
            internal_x = op.y
        elif all(shape_y == 1):
            internal_x = op.x
            internal_y = op.y

        for indices in ([1], [2], [3, 4], [2, 3, 4], [1, 2, 3, 4]):
            if indices == [1, 2, 3, 4] and mode == "multiply":
                # INTERNAL_MUL_XYKN not implemented
                continue
            if all(shape_x[indices] == shape_y[indices]):
                if all([True if i in indices else s == 1 for i, s in enumerate(shape_x)]):
                    internal_y = op.x
                    internal_x = op.y
                    break
                if all([True if i in indices else s == 1 for i, s in enumerate(shape_y)]):
                    internal_x = op.x
                    internal_y = op.y
                    break

        if internal_x is not None:
            if mode in {"add", "multiply"}:
                builder.add_elementwise(
                    name=name,
                    input_names=make_input(const_context, builder, [internal_x, internal_y]),
                    output_name=output_name,
                    mode=mode.upper(),
                )
            elif mode == "subtract":
                builder.add_activation(
                    name="_neg_y_" + name,
                    input_name=make_input(const_context, builder, op.y),
                    output_name="_neg_y_" + output_name,
                    non_linearity="LINEAR",
                    params=[-1, 0])
                if op.x == internal_y:
                    internal_x = "_neg_y_" + output_name
                else:
                    internal_y = "_neg_y_" + output_name
                builder.add_elementwise(
                    name=name,
                    input_names=make_input(const_context, builder, [internal_x, internal_y]),
                    output_name=output_name,
                    mode="ADD",
                )
            return

    if mode in {"add", "multiply", "max", "min"}:
        add_func = getattr(builder, "add_" + mode + "_broadcastable", None)

        if add_func is None:
            msg = "Element-wise binary method {} not found in builder."
            raise ValueError(msg.format(mode))

        add_func(
            name=name,
            input_names=make_input(const_context, builder, [op.x, op.y]),
            output_name=output_name,
            **kwargs
        )
    else:
        if mode in ["divide", "floor_div", "mod", "pow", "subtract"]:
            add_func = getattr(builder, "add_" + mode + "_broadcastable", None)
        elif mode == "less_equal":
            add_func = builder.add_less_than
            kwargs["use_less_than_equal"] = True
        elif mode == "greater_equal":
            add_func = builder.add_greater_than
            kwargs["use_greater_than_equal"] = True
        else:
            add_func = getattr(builder, "add_" + mode, None)

        if add_func is None:
            msg = "Element-wise binary method {} not found in builder."
            raise ValueError(msg.format(mode))

        add_func(
            name=name,
            input_names=make_input(const_context, builder, [op.x, op.y]),
            output_name=output_name,
            **kwargs
        )


def _add_logical(const_context, builder, op, mode):
    input_names = []
    input_names.append(make_input(const_context, builder, op.x))
    if mode != "NOT":
        input_names.append(make_input(const_context, builder, op.y))

    builder.add_logical(
        name=op.name, input_names=input_names, output_name=op.outputs[0].name, mode=mode
    )


@register_mil_to_nn_mapping
def abs(const_context, builder, op):
    _add_elementwise_unary(const_context, builder, op, "abs")


@register_mil_to_nn_mapping
def acos(const_context, builder, op):
    _add_elementwise_unary(const_context, builder, op, "acos")


@register_mil_to_nn_mapping
def add(const_context, builder, op):
    _add_elementwise_binary(const_context, builder, op, "add")


@register_mil_to_nn_mapping
def asin(const_context, builder, op):
    _add_elementwise_unary(const_context, builder, op, "asin")


@register_mil_to_nn_mapping
def atan(const_context, builder, op):
    _add_elementwise_unary(const_context, builder, op, "atan")


@register_mil_to_nn_mapping
def atanh(const_context, builder, op):
    _add_elementwise_unary(const_context, builder, op, "atanh")


@register_mil_to_nn_mapping
def cast(const_context, builder, op):
    if op.dtype.val in ["int32", "int64"]:
        _add_elementwise_unary(
            const_context, builder, op, "floor", output_name=op.name + "_floor"
        )
        _add_elementwise_unary(
            const_context, builder, op, "ceil", output_name=op.name + "_ceil"
        )

        builder.add_greater_than(
            name=op.name + "_cond",
            input_names=[make_input(const_context, builder, op.x)],
            output_name=op.name + "_cond",
            alpha=0.0,
        )

        builder.add_where_broadcastable(
            name=op.name,
            input_names=[op.name + i for i in ["_cond", "_floor", "_ceil"]],
            output_name=op.outputs[0].name,
        )
    elif op.dtype.val in ["fp16", "fp32", "fp64"]:
        builder.add_activation(
            name=op.name,
            non_linearity="LINEAR",
            input_name=make_input(const_context, builder, op.x),
            output_name=op.outputs[0].name,
            params=[1.0, 0.0],
        )
    elif op.dtype.val == "bool":
        builder.add_not_equal(
            name=op.name,
            input_names=op.x.name,
            output_name=op.outputs[0].name,
            alpha=0.0,
        )
    else:
        raise NotImplementedError(
            "Parameter dtype of the cast operation can be one of the {}. "
            "Provided {}".format(["int32", "int64", "fp16", "fp32", "fp64"], op.dtype.val)
        )


@register_mil_to_nn_mapping
def ceil(const_context, builder, op):
    _add_elementwise_unary(const_context, builder, op, "ceil")


@register_mil_to_nn_mapping
def clip(const_context, builder, op):
    _add_elementwise_unary(
        const_context,
        builder,
        op,
        "clip",
        min_value=op.alpha.val,
        max_value=op.beta.val,
    )


@register_mil_to_nn_mapping
def cos(const_context, builder, op):
    _add_elementwise_unary(const_context, builder, op, "cos")


@register_mil_to_nn_mapping
def cosh(const_context, builder, op):
    _add_elementwise_unary(const_context, builder, op, "cosh")

@register_mil_to_nn_mapping
def einsum(const_context, builder, op):
    '''
    MIL einsum is either
    - (B,C,H,W1) * (B,W1,H,W2) = (B,C,H,W2)
    or
    - (C,H,W1) * (W1,H,W2) = (C,H,W2)

    Hence to support it, first transpose the 2 inputs, so that the matrices
    to be multiplied are on the last 2 axes,
    then call bmm, and finally transpose the result again
    '''
    rank = op.values[0].rank
    perm = [0, 2, 1, 3] if rank == 4 else [1, 0, 2]
    input_names = make_input(const_context, builder, op.values)

    builder.add_transpose(name=op.name + "_transpose_x",
                          axes=perm,
                          input_name=input_names[0],
                          output_name=input_names[0] + "_transposed"
    )
    builder.add_transpose(name=op.name + "_transpose_y",
                          axes=perm,
                          input_name=input_names[1],
                          output_name=input_names[1] + "_transposed"
    )
    builder.add_batched_mat_mul(
        name=op.name + "_batch_matmul",
        input_names=[input_names[0] + "_transposed", input_names[1] + "_transposed"],
        output_name=op.outputs[0].name + "_pre_transpose"
    )
    builder.add_transpose(name=op.name,
                          axes=perm,
                          input_name=op.outputs[0].name + "_pre_transpose",
                          output_name=op.outputs[0].name
    )


@register_mil_to_nn_mapping
def equal(const_context, builder, op):
    _add_elementwise_binary(const_context, builder, op, "equal")


@register_mil_to_nn_mapping
def exp(const_context, builder, op):
    _add_elementwise_unary(const_context, builder, op, "exp")


@register_mil_to_nn_mapping
def exp2(const_context, builder, op):
    _add_elementwise_unary(const_context, builder, op, "exp2")


@register_mil_to_nn_mapping
def floor(const_context, builder, op):
    _add_elementwise_unary(const_context, builder, op, "floor")


@register_mil_to_nn_mapping
def floor_div(const_context, builder, op):
    _add_elementwise_binary(const_context, builder, op, "floor_div")


@register_mil_to_nn_mapping
def greater(const_context, builder, op):
    _add_elementwise_binary(const_context, builder, op, "greater_than")


@register_mil_to_nn_mapping
def greater_equal(const_context, builder, op):
    _add_elementwise_binary(const_context, builder, op, "greater_equal")


@register_mil_to_nn_mapping
def inverse(const_context, builder, op):
    _add_elementwise_unary(const_context, builder, op, "inverse", epsilon=op.epsilon.val)


@register_mil_to_nn_mapping
def less(const_context, builder, op):
    _add_elementwise_binary(const_context, builder, op, "less_than")


@register_mil_to_nn_mapping
def less_equal(const_context, builder, op):
    _add_elementwise_binary(const_context, builder, op, "less_equal")


@register_mil_to_nn_mapping
def log(const_context, builder, op):
    _add_elementwise_unary(const_context, builder, op, "log", epsilon=op.epsilon.val)


@register_mil_to_nn_mapping
def logical_and(const_context, builder, op):
    _add_logical(const_context, builder, op, "AND")


@register_mil_to_nn_mapping
def logical_not(const_context, builder, op):
    _add_logical(const_context, builder, op, "NOT")


@register_mil_to_nn_mapping
def logical_or(const_context, builder, op):
    _add_logical(const_context, builder, op, "OR")


@register_mil_to_nn_mapping
def logical_xor(const_context, builder, op):
    _add_logical(const_context, builder, op, "XOR")


@register_mil_to_nn_mapping
def maximum(const_context, builder, op):
    _add_elementwise_binary(const_context, builder, op, "max")


@register_mil_to_nn_mapping
def minimum(const_context, builder, op):
    _add_elementwise_binary(const_context, builder, op, "min")


@register_mil_to_nn_mapping
def mod(const_context, builder, op):
    _add_elementwise_binary(const_context, builder, op, "mod")


@register_mil_to_nn_mapping
def mul(const_context, builder, op):
    _add_elementwise_binary(const_context, builder, op, "multiply")


@register_mil_to_nn_mapping
def not_equal(const_context, builder, op):
    _add_elementwise_binary(const_context, builder, op, "not_equal")


@register_mil_to_nn_mapping
def pow(const_context, builder, op):
    _add_elementwise_binary(const_context, builder, op, "pow")


@register_mil_to_nn_mapping
def real_div(const_context, builder, op):
    _add_elementwise_binary(const_context, builder, op, "divide")


@register_mil_to_nn_mapping
def round(const_context, builder, op):
    _add_elementwise_unary(const_context, builder, op, "round")


@register_mil_to_nn_mapping
def rsqrt(const_context, builder, op):
    _add_elementwise_unary(const_context, builder, op, "rsqrt", epsilon=op.epsilon.val)


@register_mil_to_nn_mapping
def sign(const_context, builder, op):
    _add_elementwise_unary(const_context, builder, op, "sign")


@register_mil_to_nn_mapping
def sin(const_context, builder, op):
    _add_elementwise_unary(const_context, builder, op, "sin")


@register_mil_to_nn_mapping
def sinh(const_context, builder, op):
    _add_elementwise_unary(const_context, builder, op, "sinh")


@register_mil_to_nn_mapping
def slice_by_index(const_context, builder, op):
    rank = op.x.rank
    stride = [1] * rank if op.stride is None else op.stride.val
    begin_mask = [False] * rank if op.begin_mask is None else op.begin_mask.val
    end_mask = [False] * rank if op.end_mask is None else op.end_mask.val
    squeeze_mask = [False] * rank if op.squeeze_mask is None else op.squeeze_mask.val

    if op.begin.val is not None and op.end.val is not None:

        # If only one dimension is sliced, we should use the slice layer instead of static_slice or dynamic_slice
        # In general, slice has a better performance.
        begin = op.begin.val
        end = op.end.val
        slice_dim = []

        for i in range(rank):
            if (not begin_mask[i] and begin[i] != 0) or \
               (not end_mask[i] and end[i] != op.x.shape[i]):
               slice_dim.append(i)


        if len(slice_dim) == 1 and not squeeze_mask[slice_dim[0]]:
            dim = slice_dim[0] - rank
            if dim in [-3, -2, -1]:
                # get the axis, only channel, width, and depth dimension are supported
                axis = None
                if dim == -1:
                    axis = "width"
                elif dim == -2:
                    axis = "height"
                elif dim == -3:
                    axis = "channel"

                start_index = 0 if begin_mask[dim] else begin[dim]
                end_index = op.x.shape[dim] if end_mask[dim] else end[dim]
                shape = op.x.shape

                if not is_symbolic(shape[dim]):
                    if start_index < 0:
                        start_index += shape[dim]

                if not is_symbolic(end_index) and start_index >= 0 and stride[dim] >= 1:
                    builder.add_slice(
                        name=op.name,
                        input_name=make_input(const_context, builder, op.x),
                        output_name=op.outputs[0].name,
                        axis=axis,
                        start_index=start_index,
                        end_index=end_index,
                        stride=stride[dim],
                    )
                    return

        # use add_slice_static
        builder.add_slice_static(
            name=op.name,
            input_name=make_input(const_context, builder, op.x),
            output_name=op.outputs[0].name,
            begin_ids=op.begin.val,
            end_ids=op.end.val,
            strides=to_py_type(stride),
            begin_masks=to_py_type(begin_mask),
            end_masks=to_py_type(end_mask),
            squeeze_masks=to_py_type(squeeze_mask),
        )
    else:
        builder.add_slice_dynamic(
            name=op.name,
            input_names=make_input(const_context, builder, [op.x, op.begin, op.end]),
            output_name=op.outputs[0].name,
            strides=to_py_type(stride),
            begin_masks=to_py_type(begin_mask),
            end_masks=to_py_type(end_mask),
            squeeze_masks=to_py_type(squeeze_mask),
        )


@register_mil_to_nn_mapping
def slice_by_size(const_context, builder, op):
    """
    If the inputs satisfy
    1. op.x has static input shape for those dimension whose size is not -1
    2. op.begin and op.size are both known during compile time
    we use add_slice_static directly

    Otherwise, build a block of ops achieving slice_by_size with dynamic input x and size.
    """

    # The static case
    if op.begin.val is not None and op.size.val is not None:
        shape = op.x.shape
        begin = op.begin.val
        size = op.size.val
        rank = op.x.rank
        end = []

        for i in range(rank):
            if size[i] == -1:
                end.append(op.x.shape[i])
            else:
                end.append(begin[i] + size[i])

        if not any_symbolic(end):
            builder.add_slice_static(
                name=op.name,
                input_name=make_input(const_context, builder, op.x),
                output_name=op.outputs[0].name,
                begin_ids=begin,
                end_ids=end,
                strides=[1] * rank,
                begin_masks=[False] * rank,
                end_masks=[False] * rank,
                squeeze_masks=[False] * rank,
            )
            return

    # The dynamic case
    # get the end_index of input x
    # for instance, x with shape [2,3,4] results in [2,3,4]
    end_index_name = op.name + "_end_index"
    builder.add_get_shape(
        name=end_index_name,
        input_name=make_input(const_context, builder, op.x),
        output_name=end_index_name,
    )

    # get the mask where size = -1
    # for instance, size = [-1,1,2] results in [1,0,0]
    const_name = op.name + "_const_name"
    add_const(const_context, builder, const_name, _np.array([-1] * op.x.rank))

    is_end_mask_name = op.name + "_is_end_mask"
    builder.add_equal(
        name=is_end_mask_name,
        input_names=make_input(const_context, builder, [const_name, op.size]),
        output_name=is_end_mask_name,
    )

    # get the mask where size != -1
    # for instance, size = [-1,1,2] results in [0,1,1]
    is_not_end_mask_name = op.name + "_is_not_end_mask"
    builder.add_not_equal(
        name=is_not_end_mask_name,
        input_names=make_input(const_context, builder, [const_name, op.size]),
        output_name=is_not_end_mask_name,
    )

    # get the end index for dimensions i where size[i] = -1
    # for size[i] != -1, just make it 0
    # for instance, x with shape [2,3,4] and size = [-1,1,2]
    # results in [2,0,0]
    end_index_with_mask_name = op.name + "_end_index_with_mask"
    builder.add_elementwise(
        name=end_index_with_mask_name,
        input_names=[end_index_name, is_end_mask_name],
        output_name=end_index_with_mask_name,
        mode="MULTIPLY",
    )

    # get the end index for dimension i where size[i] != -1
    # for size[i] = 1, just make it 0
    # for instance, x with shape [2,3,4], size = [-1,1,2],
    # begin = [0,1,1] results in [0,2,3]
    end_ids = op.name + "_end_ids"
    builder.add_elementwise(
        name=end_ids,
        input_names=make_input(const_context, builder, [op.begin, op.size]),
        output_name=end_ids,
        mode="ADD",
    )

    end_index_without_mask_name = op.name + "_end_index_without_mask"
    builder.add_elementwise(
        name=end_index_without_mask_name,
        input_names=make_input(const_context, builder, [is_not_end_mask_name, end_ids]),
        output_name=end_index_without_mask_name,
        mode="MULTIPLY",
    )

    # add two end index array together to get the final index
    final_end_index_name = op.name + "_final_index"
    builder.add_elementwise(
        name=final_end_index_name,
        input_names=make_input(
            const_context,
            builder,
            [end_index_with_mask_name, end_index_without_mask_name],
        ),
        output_name=final_end_index_name,
        mode="ADD",
    )

    input_names = make_input(
        const_context, builder, [op.x, op.begin, final_end_index_name]
    )
    builder.add_slice_dynamic(
        name=op.name, input_names=input_names, output_name=op.outputs[0].name
    )


@register_mil_to_nn_mapping
def sqrt(const_context, builder, op):
    _add_elementwise_unary(const_context, builder, op, "sqrt")


@register_mil_to_nn_mapping
def square(const_context, builder, op):
    _add_elementwise_unary(const_context, builder, op, "power", alpha=2.0)


@register_mil_to_nn_mapping
def sub(const_context, builder, op):
    _add_elementwise_binary(const_context, builder, op, "subtract")


@register_mil_to_nn_mapping
def tan(const_context, builder, op):
    _add_elementwise_unary(const_context, builder, op, "tan")


@register_mil_to_nn_mapping
def threshold(const_context, builder, op):
    _add_elementwise_unary(const_context, builder, op, "threshold", alpha=op.alpha.val)


@register_mil_to_nn_mapping
def depth_to_space(const_context, builder, op):
    builder.add_reorganize_data(
        name=op.name,
        input_name=make_input(const_context, builder, op.x),
        output_name=op.outputs[0].name,
        mode="DEPTH_TO_SPACE",
        block_size=op.block_size.val,
    )


@register_mil_to_nn_mapping
def expand_dims(const_context, builder, op):
    builder.add_expand_dims(
        name=op.name,
        input_name=make_input(const_context, builder, op.x),
        output_name=op.outputs[0].name,
        axes=op.axes.val,
    )



@register_mil_to_nn_mapping
def fill(const_context, builder, op):
    if op.shape.val is None:
        builder.add_fill_dynamic(
            name=op.name,
            input_name=make_input(const_context, builder, op.shape),
            output_name=op.outputs[0].name,
            value=op.value.val,
        )
    else:
        builder.add_fill_static(
            name=op.name,
            output_name=op.outputs[0].name,
            output_shape=op.shape.val,
            value=op.value.val,
        )


@register_mil_to_nn_mapping
def random_bernoulli(const_context, builder, op):
    if op.shape.val is None:
        builder.add_random_bernoulli_dynamic(
            name=op.name,
            input_names=make_input(const_context, builder, [op.shape]),
            output_name=op.outputs[0].name,
            prob=op.prob.val,
            seed=op.seed.val,
        )
    else:
        builder.add_random_bernoulli_static(
            name=op.name,
            output_name=op.outputs[0].name,
            output_shape=op.shape.val,
            prob=op.prob.val,
            seed=op.seed.val,
        )


@register_mil_to_nn_mapping
def random_categorical(const_context, builder, op):
    builder.add_categorical_distribution(
        name=op.name,
        input_name=make_input(const_context, builder, op.x),
        output_name=op.outputs[0].name,
        num_samples=op.size.val,
        is_logits=(op.mode.val == "logits"),
        seed=op.seed.val,
    )


@register_mil_to_nn_mapping
def random_normal(const_context, builder, op):
    if op.shape.val is None:
        builder.add_random_normal_dynamic(
            name=op.name,
            input_names=make_input(const_context, builder, [op.shape]),
            output_name=op.outputs[0].name,
            mean=op.mean.val,
            stddev=op.stddev.val,
            seed=op.seed.val,
        )
    else:
        builder.add_random_normal_static(
            name=op.name,
            output_name=op.outputs[0].name,
            output_shape=op.shape.val,
            mean=op.mean.val,
            stddev=op.stddev.val,
            seed=op.seed.val,
        )


@register_mil_to_nn_mapping
def random_uniform(const_context, builder, op):
    if op.shape.val is None:
        builder.add_random_uniform_dynamic(
            name=op.name,
            input_names=make_input(const_context, builder, [op.shape]),
            output_name=op.outputs[0].name,
            minval=op.low.val,
            maxval=op.high.val,
            seed=op.seed.val,
        )
    else:
        builder.add_random_uniform_static(
            name=op.name,
            output_name=op.outputs[0].name,
            output_shape=op.shape.val,
            minval=op.low.val,
            maxval=op.high.val,
            seed=op.seed.val,
        )


@register_mil_to_nn_mapping
def gru(const_context, builder, op):
    make_input(const_context, builder, [op.x, op.initial_h])
    # Input shape: [b, s, I]
    input_name = op.x.name
    # Shape: [b, H]
    initial_h = op.initial_h.name

    weight_ih = op.weight_ih.val
    weight_hh = op.weight_hh.val
    b = op.bias.val if op.bias is not None else None
    direction = op.direction.val
    output_sequence = op.output_sequence.val

    # Add expand dims for input, in
    _expand_dim(builder, input_name + "_expanded", input_name, [3, 4])
    input_name += "_expanded"

    if direction not in {"forward", "reverse"}:
        raise ValueError(
            "Unknown direction {} for GRU layer. Supported are forward, reverse".format(
                direction
            )
        )

    # Expand initial_h
    _expand_dim(builder, initial_h + "_expanded", initial_h, [0, 3, 4])
    initial_h += "_expanded"

    def roz_to_zro(x):
        if x is None:
            return None
        r, o, z = _split(x, sections=3, axis=0)
        return [z, r, o]

    # w_x: [H*I, H*I, H*I]
    # w_h: [H*H, H*H, H*H]
    # where, format is [Z, R, O]
    # Z: Update gate, R: Reset gate, O: Output gate
    w_x = roz_to_zro(weight_ih)
    w_h = roz_to_zro(weight_hh)
    # bias format: [3*H]
    b = roz_to_zro(b)

    input_size = w_x[0].shape[1]
    hidden_size = w_x[0].shape[0]

    # 2 outputs
    # Y  : [s/1, b, h, 1, 1]
    # Y_h: [  1, b, h, 1, 1]
    output_names = [_output.name + "_5d" for _output in op.outputs]
    builder.add_gru(
        name=op.name,
        W_h=w_h,
        W_x=w_x,
        b=b,
        hidden_size=hidden_size,
        input_size=input_size,
        input_names=[input_name, initial_h],
        output_names=output_names,
        inner_activation=op.recurrent_activation.val,
        activation=op.activation.val,
        output_all=output_sequence,
        reverse_input=(direction == "reverse"),
    )

    # Squeeze Output
    # to output shape of [Seq Len or 1, Batch Size, Hidden Size]
    _squeeze(builder, op.outputs[0].name, output_names[0], axes=[3, 4])
    # Squeeze Output H and Output C
    # to output shape of [Batch Size, Hidden Size]
    _squeeze(builder, op.outputs[1].name, output_names[1], axes=[0, 3, 4])


@register_mil_to_nn_mapping
def squeeze(const_context, builder, op):
    axes = op.axes.val if op.axes is not None else None
    builder.add_squeeze(
        name=op.name,
        input_name=make_input(const_context, builder, op.x),
        output_name=op.outputs[0].name,
        axes=axes,
        squeeze_all=axes is None,
    )


@register_mil_to_nn_mapping
def topk(const_context, builder, op):
    builder.add_topk(
        name=op.name,
        input_names=make_input(const_context, builder, [op.x]),
        output_names=[output.name for output in op.outputs],
        k=op.k.val,
        axis=op.axis.val,
        use_bottom_k=op.ascending.val,
    )


@register_mil_to_nn_mapping
def l2_pool(const_context, builder, op):
    _convert_pool(const_context=const_context, builder=builder, op=op, mode="l2")


@register_mil_to_nn_mapping
def linear(const_context, builder, op):
    out_channels, in_channels = op.weight.shape
    if op.x.rank and op.x.rank <= 3 and op.x.rank > 0:
        has_bias = op.bias is not None and op.bias.val is not None
        builder.add_inner_product(
            name=op.name,
            W=op.weight.val,
            b=op.bias.val if has_bias else None,
            input_channels=in_channels,
            output_channels=out_channels,
            has_bias=has_bias,
            input_name=make_input(const_context, builder, op.x),
            output_name=op.outputs[0].name,
        )
    else:
        builder.add_batched_mat_mul(
            name=op.name,
            input_names=make_input(const_context, builder, [op.x]),
            output_name=op.outputs[0].name,
            W=op.weight.val.T,
            bias=op.bias.val,
            weight_matrix_rows=in_channels,
            weight_matrix_columns=out_channels,
        )

@register_mil_to_nn_mapping
def matmul(const_context, builder, op):
    weight = None
    rows, columns = 0, 0

    if (
        op.y.val is not None
        and op.y.rank == 2
        and len(op.y.child_ops) == 1
        and len(op.y.consuming_blocks) == 0
    ):

        weight = op.y.val
        if op.transpose_y.val:
            weight = weight.transpose((1, 0))

        rows, columns = weight.shape
        input_names = make_input(const_context, builder, [op.x])

        if op.transpose_x.val:
            perm = [i for i in range(op.x.rank)]
            perm[-1], perm[-2] = perm[-2], perm[-1]
            name = op.name + "_x_transpose"
            builder.add_transpose(
                name=name, axes=perm, input_name=input_names[0], output_name=name
            )
            input_names = [name]

    else:
        input_names = make_input(const_context, builder, [op.x, op.y])

    builder.add_batched_mat_mul(
        name=op.name,
        input_names=input_names,
        output_name=op.outputs[0].name,
        transpose_a=op.transpose_x.val,
        transpose_b=op.transpose_y.val,
        W=weight,
        weight_matrix_rows=rows,
        weight_matrix_columns=columns,
    )


@register_mil_to_nn_mapping
def max_pool(const_context, builder, op):
    _convert_pool(const_context=const_context, builder=builder, op=op, mode="max")


@register_mil_to_nn_mapping
def non_zero(const_context, builder, op):
    builder.add_where_nonzero(
        name=op.name,
        input_name=make_input(const_context, builder, op.x),
        output_name=op.outputs[0].name,
    )


@register_mil_to_nn_mapping
def lstm(const_context, builder, op):
    make_input(const_context, builder, [op.x, op.initial_h, op.initial_c])
    # Input shape [b, s, I]
    input_name = op.x.name
    # Shape: [b, DIRECTION*H]
    initial_h = op.initial_h.name
    initial_c = op.initial_c.name

    wt_ih = op.weight_ih.val
    wt_hh = op.weight_hh.val
    b = op.bias.val if op.bias is not None else None
    direction = op.direction.val
    output_sequence = op.output_sequence.val
    peephole = op.peephole.val if op.peephole is not None else None
    # High enough clip value to be ineffective!
    clip = 500.0 if op.clip is None else op.clip.val

    # Add expand dims for input, in
    _expand_dim(builder, input_name + "_expanded", input_name, [3, 4])
    input_name += "_expanded"

    if direction in {"forward", "reverse"}:
        # Expand initial_h and initial_c,
        # from shape (B, H) to shape (1, Batch, H, 1, 1)
        _expand_dim(builder, initial_h + "_expanded", initial_h, [0, 3, 4])
        initial_h += "_expanded"
        # initial_h may have the same name as initial_c (e.g., same Var).
        # Append a different string to avoid conflict
        _expand_dim(builder, initial_c + "_expanded2", initial_c, [0, 3, 4])
        initial_c += "_expanded2"

        # w_x: [H*I, H*I, H*I, H*I]
        # w_h: [H*H, H*H, H*H, H*H]
        # where format is, [input gate, forget gate, output gate, cell gate]
        w_x = _split(wt_ih, sections=4)
        w_h = _split(wt_hh, sections=4)
        # bias format: [4*H]
        b = _split(b, sections=4)  # ifoz layout
        # peephole format: [3*H]
        # where format is, [input gate, forget gate, output gate]
        peephole = _split(peephole, sections=3)

        input_size = w_x[0].shape[1]
        hidden_size = w_h[0].shape[1]

        # 3 outputs
        # Y  : [s/1, b, h, 1, 1]
        # Y_h: [  1, b, h, 1, 1]
        # Y_c: [  1, b, h, 1, 1]
        output_names = [_output.name + "_5d" for _output in op.outputs]
        builder.add_unilstm(
            name=op.name,
            W_h=w_h,
            W_x=w_x,
            b=b,
            hidden_size=hidden_size,
            input_size=input_size,
            input_names=[input_name, initial_h, initial_c],
            output_names=output_names,
            inner_activation=op.recurrent_activation.val,
            cell_state_update_activation=op.cell_activation.val,
            output_activation=op.activation.val,
            peep=peephole,
            output_all=output_sequence,
            cell_clip_threshold=clip,
            reverse_input=(direction == "reverse"),
        )

        # Squeeze Output
        # to output shape of [Seq Len or 1, Batch Size, Hidden Size]
        _squeeze(builder, op.outputs[0].name, output_names[0], axes=[3, 4])
        # Squeeze Output H and Output C
        # to output shape of [Batch Size, Hidden Size]
        _squeeze(builder, op.outputs[1].name, output_names[1], axes=[0, 3, 4])
        _squeeze(builder, op.outputs[2].name, output_names[2], axes=[0, 3, 4])

    elif direction == "bidirectional":
        # Expand initial_h and initial_c
        # Issue #810
        num_layer = len(builder.layers)
        initial_h_expand = initial_h + "_expanded" + "_" + str(num_layer)
        # from shape (B, 2*H) to shape (1, Batch, 2*H, 1, 1)
        if not (initial_h_expand in set(builder.layers)):
            _expand_dim(builder, initial_h_expand, initial_h, [0, 3, 4])
        initial_h = initial_h_expand

        # initial_h may have the same name as initial_c (e.g., same Var)
        initial_c_expand = initial_c + "_expanded2" + "_" + str(num_layer)
        if not (initial_c_expand in set(builder.layers)):
            _expand_dim(builder, initial_c_expand, initial_c, [0, 3, 4])
        initial_c = initial_c_expand

        initial_h_f = initial_h + "_forward"
        initial_h_r = initial_h + "_reverse"
        initial_c_f = initial_c + "_forward"
        initial_c_r = initial_c + "_reverse"

        # split input_h and input_c into two parts
        builder.add_split_nd(
            name=op.name + "_split_h",
            input_name=initial_h,
            output_names=[initial_h_f, initial_h_r],
            axis=2,
        )
        builder.add_split_nd(
            name=op.name + "_split_c",
            input_name=initial_c,
            output_names=[initial_c_f, initial_c_r],
            axis=2,
        )

        wt_ih_back = op.weight_ih_back.val
        wt_hh_back = op.weight_hh_back.val
        # Get weights here
        # weight format: [I+H, 2*4*H] -> [I+H, 4*H (forward):4*H (backward)]
        hidden_size = wt_hh.shape[1]
        input_size = wt_ih.shape[1]

        # f_w_x and r_w_x: [H*I, H*I, H*I, H*I]
        # f_w_h and r_w_h: [H*H, H*H, H*H, H*H]
        # where format is, [input gate, forget gate, output gate, cell gate]
        w_x = _split(wt_ih, sections=4)
        w_h = _split(wt_hh, sections=4)
        r_w_x = _split(wt_ih_back, sections=4)
        r_w_h = _split(wt_hh_back, sections=4)

        # f_b and r_b format: [4*H]
        b_back = op.bias_back.val if op.bias_back is not None else None
        f_b, r_b = None, None
        if b is not None:
            f_b = _split(b, sections=4)
        if b_back is not None:
            r_b = _split(b_back, sections=4)

        # peephole format: [2*3*H] -> [3*H (forward) : 3*H (backward)]
        peephole_back = op.peephole_back.val if op.peephole_back is not None else None
        f_peephole, r_peephole = None, None
        if peephole is not None:
            f_peephole = _split(peephole, sections=3)
        if peephole_back is not None:
            r_peephole = _split(peephole_back, sections=3)

        output_names = [
            op.outputs[0].name + "_5d",  # Output Y           [s/1, b, 2*h, 1, 1]
            op.outputs[1].name + "_5d_foward",  # Output Y_h         [  1, b,   h, 1, 1]
            op.outputs[2].name
            + "_5d_forward",  # Output Y_c         [  1, b,   h, 1, 1]
            op.outputs[1].name
            + "_5d_reverse",  # Output Y_h_reverse [  1, b,   h, 1, 1]
            op.outputs[2].name + "_5d_reverse",
        ]  # Output Y_c_reverse [  1, b,   h, 1, 1]

        builder.add_bidirlstm(
            name=op.name,
            W_h=w_h,
            W_x=w_x,
            b=f_b,
            W_h_back=r_w_h,
            W_x_back=r_w_x,
            b_back=r_b,
            hidden_size=hidden_size,
            input_size=input_size,
            input_names=[
                input_name,
                initial_h_f,
                initial_c_f,
                initial_h_r,
                initial_c_r,
            ],
            output_names=output_names,
            inner_activation=op.recurrent_activation.val,
            cell_state_update_activation=op.cell_activation.val,
            output_activation=op.activation.val,
            peep=f_peephole,
            peep_back=r_peephole,
            output_all=output_sequence,
            cell_clip_threshold=clip,
        )

        # Squeeze Output
        # to output shape of [Seq Len or 1, Batch Size, 2*Hidden Size]
        _squeeze(builder, op.outputs[0].name, output_names[0], axes=[3, 4])

        # Output H is of format
        # 1, Batch_Size, Hidden_Size, 1, 1
        # Concat to make it
        # 1, Batch_Size, 2*Hidden_Size, 1, 1
        builder.add_elementwise(
            name=op.outputs[1].name + "_5d",
            input_names=[output_names[1], output_names[3]],
            output_name=op.outputs[1].name + "_5d",
            mode="CONCAT",
        )
        # Output C is of format
        # 1, Batch_Size, Hidden_Size, 1, 1
        builder.add_elementwise(
            name=op.outputs[2].name + "_5d",
            input_names=[output_names[2], output_names[4]],
            output_name=op.outputs[2].name + "_5d",
            mode="CONCAT",
        )

        # Squeeze Output H and Output C
        # to output shape of [Batch Size, 2*Hidden Size]
        _squeeze(
            builder, op.outputs[1].name, op.outputs[1].name + "_5d", axes=[0, 3, 4]
        )
        _squeeze(
            builder, op.outputs[2].name, op.outputs[2].name + "_5d", axes=[0, 3, 4]
        )
    else:
        raise ValueError(
            "Unknown direction {} for LSTM layer. Supported are forward, reverse or bidirectional".format(
                direction
            )
        )


@register_mil_to_nn_mapping
def reshape(const_context, builder, op):
    if op.shape.val is None:
        builder.add_reshape_dynamic(
            name=op.name,
            input_names=make_input(const_context, builder, [op.x, op.shape]),
            output_name=op.outputs[0].name,
        )
    elif -1 in op.shape.val and len(op.shape.val) == op.x.rank:
        # Support 0 in shape.
        builder.add_rank_preserving_reshape(
            name=op.name,
            input_name=make_input(const_context, builder, op.x),
            output_name=op.outputs[0].name,
            output_shape=op.shape.val,
        )
    else:
        if 0 in op.shape.val:
            # Does not support 0 in shape
            msg = "Use 0 in shape only if len(shape) == x.rank. Report bug."
            raise ValueError(msg)
        output_shape = (1,) if len(op.shape.val) == 0 or 0 in op.shape.shape else op.shape.val
        builder.add_reshape_static(
            name=op.name,
            input_name=make_input(const_context, builder, op.x),
            output_name=op.outputs[0].name,
            output_shape=output_shape,
        )


@register_mil_to_nn_mapping
def reduce_argmax(const_context, builder, op):
    builder.add_argmax(
        name=op.name,
        input_name=make_input(const_context, builder, op.x),
        output_name=op.outputs[0].name,
        axis=op.axis.val,
        keepdims=op.keep_dims.val,
    )


@register_mil_to_nn_mapping
def reduce_argmin(const_context, builder, op):
    builder.add_argmin(
        name=op.name,
        input_name=make_input(const_context, builder, op.x),
        output_name=op.outputs[0].name,
        axis=op.axis.val,
        keepdims=op.keep_dims.val,
    )


def _reduce_axes(const_context, builder, builder_op, op):
    axes = op.axes.val if op.axes is not None else op.axes
    builder_op(
        name=op.name,
        input_name=make_input(const_context, builder, op.x),
        output_name=op.outputs[0].name,
        axes=axes,
        keepdims=op.keep_dims.val,
        reduce_all=axes is None,
    )


@register_mil_to_nn_mapping
def reduce_l1_norm(const_context, builder, op):
    _reduce_axes(const_context, builder, builder.add_reduce_l1, op)


@register_mil_to_nn_mapping
def reduce_l2_norm(const_context, builder, op):
    _reduce_axes(const_context, builder, builder.add_reduce_l2, op)


@register_mil_to_nn_mapping
def reduce_log_sum(const_context, builder, op):
    _reduce_axes(const_context, builder, builder.add_reduce_logsum, op)


@register_mil_to_nn_mapping
def reduce_log_sum_exp(const_context, builder, op):
    _reduce_axes(const_context, builder, builder.add_reduce_logsumexp, op)


@register_mil_to_nn_mapping
def reduce_max(const_context, builder, op):
    if not _try_convert_global_pool(const_context, builder, op, mode="max"):
        _reduce_axes(const_context, builder, builder.add_reduce_max, op)


@register_mil_to_nn_mapping
def reduce_mean(const_context, builder, op):
    if not _try_convert_global_pool(const_context, builder, op, mode="average"):
        _reduce_axes(const_context, builder, builder.add_reduce_mean, op)


@register_mil_to_nn_mapping
def reduce_min(const_context, builder, op):
    _reduce_axes(const_context, builder, builder.add_reduce_min, op)


@register_mil_to_nn_mapping
def reduce_prod(const_context, builder, op):
    _reduce_axes(const_context, builder, builder.add_reduce_prod, op)


@register_mil_to_nn_mapping
def reduce_sum(const_context, builder, op):
    _reduce_axes(const_context, builder, builder.add_reduce_sum, op)


@register_mil_to_nn_mapping
def reduce_sum_square(const_context, builder, op):
    _reduce_axes(const_context, builder, builder.add_reduce_sumsquare, op)


@register_mil_to_nn_mapping
def reverse(const_context, builder, op):
    reverse_dim = [False] * op.x.rank
    if op.axes is None:
        reverse_dim = [True] * op.x.rank
    else:
        for axis in op.axes.val:
            reverse_dim[axis] = True
    builder.add_reverse(
        name=op.name,
        input_name=make_input(const_context, builder, op.x),
        output_name=op.outputs[0].name,
        reverse_dim=reverse_dim,
    )


@register_mil_to_nn_mapping
def reverse_sequence(const_context, builder, op):
    builder.add_reverse_sequence(
        name=op.name,
        input_names=make_input(const_context, builder, [op.x, op.lengths]),
        output_name=op.outputs[0].name,
        batch_axis=op.batch_axis.val,
        seq_axis=op.seq_axis.val,
    )


@register_mil_to_nn_mapping
def rnn(const_context, builder, op):
    input_name = make_input(const_context, builder, op.x)  # [b, s, I]
    initial_h = make_input(const_context, builder, op.initial_h)  # [b, H]

    w_ih = op.weight_ih.val
    w_hh = op.weight_hh.val
    b = op.bias.val if op.bias is not None else None
    direction = op.direction.val
    output_sequence = op.output_sequence.val
    activation = op.activation.val

    # Add expand dims for input, in
    _expand_dim(builder, input_name + "_expanded", input_name, [3, 4])
    input_name += "_expanded"

    if direction not in {"forward", "reverse"}:
        raise ValueError(
            "Unknown direction {} for RNN layer. Supported are forward and reverse".format(
                direction
            )
        )

    # Expand initial_h and initial_c
    _expand_dim(builder, initial_h + "_expanded", initial_h, [2, 3, 4])
    initial_h += "_expanded"

    # w_x: (H, I)
    # w_h: (H, H)
    hidden_size = w_hh.shape[0]
    input_size = w_ih.shape[-1]

    # 3 outputs
    # Y  : [s/1, b, h, 1, 1]
    # Y_h: [  1, b, h, 1, 1]
    output_names = [_output.name + "_5d" for _output in op.outputs]
    builder.add_simple_rnn(
        name=op.name,
        W_h=w_hh,
        W_x=w_ih,
        b=b,
        hidden_size=hidden_size,
        input_size=input_size,
        input_names=[input_name, initial_h],
        output_names=output_names,
        activation=activation,
        output_all=output_sequence,
        reverse_input=(direction == "reverse"),
    )

    # Squeeze Output
    # to output shape of [Seq Len or 1, Batch Size, Hidden Size]
    _squeeze(builder, op.outputs[0].name, output_names[0], [3, 4])
    # Squeeze Output H and Output C
    # to output shape of [Batch Size, Hidden Size]
    _squeeze(builder, op.outputs[1].name, output_names[1], [0, 3, 4])


@register_mil_to_nn_mapping
def select(const_context, builder, op):
    builder.add_where_broadcastable(
        name=op.name,
        input_names=make_input(const_context, builder, [op.cond, op.a, op.b]),
        output_name=op.outputs[0].name,
    )


@register_mil_to_nn_mapping
def space_to_depth(const_context, builder, op):
    builder.add_reorganize_data(
        name=op.name,
        input_name=make_input(const_context, builder, op.x),
        output_name=op.outputs[0].name,
        mode="SPACE_TO_DEPTH",
        block_size=op.block_size.val,
    )


@register_mil_to_nn_mapping
def transpose(const_context, builder, op):
    builder.add_transpose(
        name=op.name,
        axes=op.perm.val,
        input_name=make_input(const_context, builder, op.x),
        output_name=op.outputs[0].name,
    )


@register_mil_to_nn_mapping
def gather(const_context, builder, op):
    is_embedding = False

    if op.x.val is not None:
        W = op.x.val
        if len(W.shape) == 2:
            if op.axis.val == 0 or op.axis.val == -2:
                if len(op.x.child_ops) == 1:
                    # the constant feeding into the gather doesn't go to any other op
                    is_embedding = True

    if is_embedding:
        """"
        The following:
            %3 = gather(%1, %2, axis=0) # %1 is a constant matrix of shape (vocab_size, embedding_size)
        can be mapped to:
            %2_e = expand_dims(%2, axis=-1)
            %3 = embeddingND(%2_e, weight=%1)
        """
        builder.add_expand_dims(
            name=op.name + "_expand_dims",
            input_name=make_input(const_context, builder, op.indices),
            output_name=op.name + "_expand_dims",
            axes=[-1],
        )

        builder.add_embedding_nd(
            name=op.name,
            input_name=op.name + "_expand_dims",
            output_name=op.outputs[0].name,
            vocab_size=W.shape[0],
            embedding_size=W.shape[1],
            W=_np.transpose(W),
        )

    else:
        builder.add_gather(
            name=op.name,
            input_names=make_input(const_context, builder, [op.x, op.indices]),
            output_name=op.outputs[0].name,
            axis=op.axis.val,
        )


@register_mil_to_nn_mapping
def scatter(const_context, builder, op):
    builder.add_scatter(
        name=op.name,
        input_names=make_input(
            const_context, builder, [op.data, op.indices, op.updates]
        ),
        output_name=op.outputs[0].name,
        axis=op.axis.val,
        mode=op.mode.val.upper(),
    )


@register_mil_to_nn_mapping
def gather_along_axis(const_context, builder, op):
    builder.add_gather_along_axis(
        name=op.name,
        input_names=make_input(const_context, builder, [op.x, op.indices]),
        output_name=op.outputs[0].name,
        axis=op.axis.val,
    )


@register_mil_to_nn_mapping
def scatter_along_axis(const_context, builder, op):
    builder.add_scatter_along_axis(
        name=op.name,
        input_names=make_input(
            const_context, builder, [op.data, op.indices, op.updates]
        ),
        output_name=op.outputs[0].name,
        axis=op.axis.val,
        mode=op.mode.val.upper(),
    )


@register_mil_to_nn_mapping
def gather_nd(const_context, builder, op):
    builder.add_gather_nd(
        name=op.name,
        input_names=make_input(
            const_context, builder, [op.x, op.indices]
        ),
        output_name=op.outputs[0].name,
    )


@register_mil_to_nn_mapping
def scatter_nd(const_context, builder, op):
    builder.add_scatter_nd(
        name=op.name,
        input_names=make_input(
            const_context, builder, [op.data, op.indices, op.updates],
        ),
        output_name=op.outputs[0].name,
        mode=op.mode.val.upper(),
    )

@register_mil_to_nn_mapping
def silu(const_context, builder, op):
    '''
    silu is:
    y = x * sigmoid(x)
    '''
    inp = make_input(const_context, builder, op.x)
    builder.add_activation(
        name=op.name + "__silu_sigmoid__",
        non_linearity="SIGMOID",
        input_name=inp,
        output_name=op.name + "__silu_sigmoid__",
    )
    builder.add_elementwise(
        name=op.name,
        input_names=[inp, op.name + "__silu_sigmoid__"],
        output_name=op.outputs[0].name,
        mode='MULTIPLY',
    )


@register_mil_to_nn_mapping
def tile(const_context, builder, op):
    inputs = [make_input(const_context, builder, op.x)]
    if op.reps.val is None:
        inputs.append(op.reps.name)
    builder.add_tile(
        name=op.name,
        reps=op.reps.val,
        input_name=inputs,
        output_name=op.outputs[0].name,
    )


@register_mil_to_nn_mapping
def tanh(const_context, builder, op):
    builder.add_activation(
        name=op.name,
        non_linearity="TANH",
        input_name=make_input(const_context, builder, op.x),
        output_name=op.outputs[0].name,
    )


@register_mil_to_nn_mapping
def scaled_tanh(const_context, builder, op):
    builder.add_activation(
        name=op.name,
        non_linearity="SCALED_TANH",
        input_name=make_input(const_context, builder, op.x),
        output_name=op.outputs[0].name,
        params=[op.alpha.val, op.beta.val],
    )


@register_mil_to_nn_mapping
def sigmoid(const_context, builder, op):
    builder.add_activation(
        name=op.name,
        non_linearity="SIGMOID",
        input_name=make_input(const_context, builder, op.x),
        output_name=op.outputs[0].name,
    )


@register_mil_to_nn_mapping
def sigmoid_hard(const_context, builder, op):
    builder.add_activation(
        name=op.name,
        non_linearity="SIGMOID_HARD",
        input_name=make_input(const_context, builder, op.x),
        output_name=op.outputs[0].name,
        params=[op.alpha.val, op.beta.val],
    )


@register_mil_to_nn_mapping
def erf(const_context, builder, op):
    builder.add_erf(
        name=op.name,
        input_name=make_input(const_context, builder, op.x),
        output_name=op.outputs[0].name,
    )


@register_mil_to_nn_mapping
def thresholded_relu(const_context, builder, op):
    builder.add_activation(
        name=op.name,
        non_linearity="THRESHOLDEDRELU",
        input_name=make_input(const_context, builder, op.x),
        output_name=op.outputs[0].name,
        params=op.alpha.val,
    )


@register_mil_to_nn_mapping
def elu(const_context, builder, op):
    builder.add_activation(
        name=op.name,
        non_linearity="ELU",
        input_name=make_input(const_context, builder, op.x),
        output_name=op.outputs[0].name,
        params=op.alpha.val,
    )


@register_mil_to_nn_mapping
def leaky_relu(const_context, builder, op):
    builder.add_activation(
        name=op.name,
        non_linearity="LEAKYRELU",
        input_name=make_input(const_context, builder, op.x),
        output_name=op.outputs[0].name,
        params=[op.alpha.val],
    )


@register_mil_to_nn_mapping
def gelu(const_context, builder, op):
    builder.add_gelu(
        name=op.name,
        input_name=make_input(const_context, builder, op.x),
        output_name=op.outputs[0].name,
        mode=op.mode.val,
    )


@register_mil_to_nn_mapping
def softplus(const_context, builder, op):
    builder.add_activation(
        name=op.name,
        non_linearity="SOFTPLUS",
        input_name=make_input(const_context, builder, op.x),
        output_name=op.outputs[0].name,
    )


@register_mil_to_nn_mapping
def softmax(const_context, builder, op):
    rank = op.x.rank
    if op.axis.val == -3 or op.axis.val > 0 and op.axis.val == rank - 3:
        builder.add_softmax(
            name=op.name, input_name=op.x.name, output_name=op.outputs[0].name,
        )
    else:
        builder.add_softmax_nd(
            name=op.name,
            input_name=op.x.name,
            output_name=op.outputs[0].name,
            axis=op.axis.val,
        )


@register_mil_to_nn_mapping
def softplus_parametric(const_context, builder, op):
    builder.add_activation(
        name=op.name,
        non_linearity="PARAMETRICSOFTPLUS",
        input_name=make_input(const_context, builder, op.x),
        input_shape=op.x.shape,
        input_rank=op.x.rank,
        output_name=op.outputs[0].name,
        params=[op.alpha.val, op.beta.val],
    )


@register_mil_to_nn_mapping
def softsign(const_context, builder, op):
    builder.add_activation(
        name=op.name,
        non_linearity="SOFTSIGN",
        input_name=make_input(const_context, builder, op.x),
        output_name=op.outputs[0].name,
    )


@register_mil_to_nn_mapping
def linear_activation(const_context, builder, op):
    builder.add_activation(
        name=op.name,
        non_linearity="LINEAR",
        input_name=make_input(const_context, builder, op.x),
        output_name=op.outputs[0].name,
        params=[op.alpha.val, op.beta.val],
    )


@register_mil_to_nn_mapping
def relu(const_context, builder, op):
    builder.add_activation(
        name=op.name,
        non_linearity="RELU",
        input_name=make_input(const_context, builder, op.x),
        output_name=op.outputs[0].name,
    )


@register_mil_to_nn_mapping
def clamped_relu(const_context, builder, op):
    builder.add_clamped_relu(
        name=op.name,
        input_name=make_input(const_context, builder, op.x),
        output_name=op.outputs[0].name,
        alpha=op.alpha.val,
        beta=op.beta.val,
    )


@register_mil_to_nn_mapping
def relu6(const_context, builder, op):
    builder.add_activation(
        name=op.name + "__relu6_relu__",
        input_name=make_input(const_context, builder, op.x),
        output_name=op.name + "__relu6_relu__",
        non_linearity="RELU",
    )
    builder.add_activation(
        name=op.name + "__relu6_neg__",
        input_name=op.name + "__relu6_relu__",
        output_name=op.name + "__relu6_neg__",
        non_linearity="LINEAR",
        params=[-1, 0],
    )
    builder.add_unary(
        name=op.name + "__relu6_threshold6__",
        input_name=op.name + "__relu6_neg__",
        output_name=op.name + "__relu6_threshold6__",
        mode="threshold",
        alpha=-6,
    )
    builder.add_activation(
        name=op.name,
        input_name=op.name + "__relu6_threshold6__",
        output_name=op.outputs[0].name,
        non_linearity="LINEAR",
        params=[-1, 0],
    )


@register_mil_to_nn_mapping
def prelu(const_context, builder, op):
    builder.add_activation(
        name=op.name,
        non_linearity="PRELU",
        input_name=make_input(const_context, builder, op.x),
        input_shape=op.x.shape,
        input_rank=op.x.rank,
        output_name=op.outputs[0].name,
        params=op.alpha.val,
    )


@register_mil_to_nn_mapping
def pad(const_context, builder, op):
    if len(op.pad.shape) != 1:
        raise ValueError("Pad should be a 1D tensor.")

    pad = op.pad.val
    mode = op.mode.val
    constant_val = op.constant_val.val

    nn_mode_mapping = {"reflect": "reflection", "replicate": "replication"}
    mode = nn_mode_mapping.get(mode, mode)

    if pad is not None:
        missing_dims = op.x.rank - len(pad) // 2
        pad = [0, 0] * missing_dims + list(pad)


    if pad is not None and op.x.rank > 1 and all(i == 0 for i in pad[:-4]):
        pad = pad[-4:]
        left, right = pad[2], pad[3]
        top, bottom = pad[0], pad[1]
        layer = builder.add_padding(
            name=op.name,
            left=left,
            right=right,
            top=top,
            bottom=bottom,
            input_name=make_input(const_context, builder, op.x),
            output_name=op.outputs[0].name,
            padding_type=mode,
            value=constant_val,
        )
    elif mode == "constant":
        if pad is None:
            builder.add_constant_pad(
                name=op.name,
                input_names=make_input(const_context, builder, [op.x, op.pad]),
                output_name=op.outputs[0].name,
                value=constant_val
            )
        else:
            builder.add_constant_pad(
                name=op.name,
                input_names=make_input(const_context, builder, [op.x]),
                output_name=op.outputs[0].name,
                value=constant_val,
                pad_amounts=pad,
            )
    else:
        raise ValueError("Unsupported mode for Pad layer! {}".format(mode))


@register_mil_to_nn_mapping
def instance_norm(const_context, builder, op):
    channels = op.x.shape[1]
    gamma = _np.array([1.0] * channels) if op.gamma is None else op.gamma.val
    beta = _np.array([0.0] * channels) if op.beta is None else op.beta.val

    x_name = make_input(const_context, builder, op.x)
    out_name = op.outputs[0].name

    if op.x.rank == 3:
        x_name = op.name + "_expanded"
        builder.add_expand_dims(
            name=x_name, input_name=op.x.name, output_name=x_name, axes=[-2],
        )
        out_name += "_instance_norm"

    builder.add_batchnorm(
        name=op.name,
        channels=channels,
        gamma=gamma,
        beta=beta,
        input_name=x_name,
        output_name=out_name,
        compute_mean_var=True,
        instance_normalization=True,
        epsilon=op.epsilon.val,
    )

    # Squeeze added `Height` dimension for 1d case
    if op.x.rank == 3:
        x_name = op.name + "_squeeze"
        builder.add_squeeze(
            name=x_name,
            input_name=out_name,
            output_name=op.outputs[0].name,
            axes=[-2],
        )




@register_mil_to_nn_mapping
def l2_norm(const_context, builder, op):
    builder.add_l2_normalize(
        name=op.name,
        input_name=make_input(const_context, builder, op.x),
        output_name=op.outputs[0].name,
        epsilon=op.epsilon.val,
    )


@register_mil_to_nn_mapping
def layer_norm(const_context, builder, op):

    rank = op.x.rank
    input_shape = [-1 if is_symbolic(dim) else dim for dim in list(op.x.shape)]
    axes = list(range(op.x.rank)) if op.axes.val is None else op.axes.val
    axes = [axis+rank if axis < 0 else axis for axis in op.axes.val]
    epsilon = op.epsilon.val

    # if input shape = (X1, X2) or (X0, X1, X2), axes = [-1], X1 and X2 are known
    # then the following operations are performed
    # - reshape to (X1, 1, X2) / (X0, X1, 1, X2)
    # - apply MVN layer, which normalizes across last 2 dims
    # - apply scale layer
    # - reshape back to (X1, X2) / (X0, X1, X2)
    # Otherwise, we express the layer_norm as primitive operations
    if rank in [2, 3] and len(axes) == 1 and axes[0] == rank - 1 and input_shape.count(-1) < 2 \
        and input_shape[-1] != -1 and input_shape[-2] != -1:

        reshaped_shape = input_shape[:]
        # Insert a singleton dimension in the 'height' position
        reshaped_shape.insert(-1, 1)

        # Scale layer can't take parameters of size [W], but can take [1, H, W], and H=1 in this case
        gamma = _np.ones((1, 1, reshaped_shape[-1])) if op.gamma is None else _np.expand_dims(op.gamma.val, axis=(0, 1))
        beta = _np.zeros((1, 1, reshaped_shape[-1])) if op.beta is None else _np.expand_dims(op.beta.val, axis=(0, 1))

        builder.add_reshape_static(
            name=op.name + "_reshape",
            input_name=make_input(const_context, builder, op.x),
            output_name=op.name + "_reshape",
            output_shape=reshaped_shape,
        )

        builder.add_mvn(
            name=op.name + "_mvn",
            input_name=op.name + "_reshape",
            output_name=op.name + "_mvn",
            across_channels=False,
            normalize_variance=True,
            epsilon=epsilon,
        )

        builder.add_scale(
            name=op.name + "_scale",
            input_name=op.name + "_mvn",
            output_name=op.name + "_scale",
            W=gamma,
            b=beta,
            has_bias=True,
            shape_scale=_np.shape(gamma),
            shape_bias=_np.shape(beta),
        )

        builder.add_reshape_static(
            name=op.name,
            input_name=op.name + "_scale",
            output_name=op.outputs[0].name,
            output_shape=input_shape,
        )

    else: # We don't meet the conditions for an MVN layer, so we use primitives
        mean_name = op.name + "_mean"
        builder.add_reduce_mean(
            name=mean_name,
            input_name=make_input(const_context, builder, op.x),
            output_name=mean_name,
            axes=axes,
            keepdims=True,
            reduce_all=False,
        )

        sub_mean_name = op.name + "_sub_mean"
        builder.add_subtract_broadcastable(
                name=sub_mean_name,
                input_names=[op.x.name, mean_name],
                output_name=sub_mean_name,
        )

        square_name = op.name + '_square'
        builder.add_unary(
                name=square_name,
                input_name=sub_mean_name,
                output_name=square_name,
                mode="power",
                alpha=2.0,
        )

        square_sum_name = op.name + '_square_sum'
        builder.add_reduce_sum(
            name=square_sum_name,
            input_name=square_name,
            output_name=square_sum_name,
            axes=axes,
            keepdims=True,
            reduce_all=False,
        )

        normalized_shape = [op.x.shape[i] if i in axes else 1 for i in range(rank)]
        if not any_symbolic(normalized_shape):
            div_prod_name = op.name + '_div_constant'
            add_const(const_context, builder, div_prod_name, _np.prod(normalized_shape))
        else:
            raise NotImplementedError("dynamic shape input nor supported for layer_norm")

        div_square_sum_name = op.name + '_div_square_sum'
        builder.add_divide_broadcastable(
            name=div_square_sum_name,
            input_names=[square_sum_name, div_prod_name],
            output_name=div_square_sum_name
        )

        epsilon_const_name = op.name + '_epsilon'
        add_const(const_context, builder, epsilon_const_name, epsilon)
        add_epsilon_name = op.name + '_add_epsilon'
        builder.add_elementwise(
            name=add_epsilon_name,
            input_names=[div_square_sum_name, epsilon_const_name],
            output_name=add_epsilon_name,
            mode="ADD",
        )

        sqrt_name = op.name + '_sqrt'
        builder.add_unary(
            name=sqrt_name,
            input_name=add_epsilon_name,
            output_name=sqrt_name,
            mode="sqrt",
        )

        div_name = op.name + '_divide'
        builder.add_divide_broadcastable(
            name=div_name,
            input_names=[sub_mean_name, sqrt_name],
            output_name=div_name
        )

        gamma = _np.ones(normalized_shape) if op.gamma is None else _np.reshape(op.gamma.val, normalized_shape)
        beta = _np.zeros(normalized_shape) if op.beta is None else _np.reshape(op.beta.val, normalized_shape)

        gamma_name = op.name + '_gamma'
        beta_name = op.name + '_beta'
        add_const(const_context, builder, gamma_name, gamma)
        add_const(const_context, builder, beta_name, beta)

        mul_name = op.name + '_mul'
        builder.add_multiply_broadcastable(
            name=mul_name,
            input_names=[div_name, gamma_name],
            output_name=mul_name,
        )

        builder.add_add_broadcastable(
            name=op.name,
            input_names=[mul_name, beta_name],
            output_name=op.outputs[0].name,
        )


@register_mil_to_nn_mapping
def local_response_norm(const_context, builder, op):
    builder.add_lrn(
        name=op.name,
        input_name=make_input(const_context, builder, op.x),
        output_name=op.outputs[0].name,
        alpha=op.alpha.val,
        beta=op.beta.val,
        local_size=op.size.val,
        k=op.k.val,
    )


@register_mil_to_nn_mapping
def conv_transpose(const_context, builder, op):
    x_name = make_input(const_context, builder, op.x)
    out_name = op.outputs[0].name

    # Special handling for 1d conv transpose
    is_conv_transpose_1d = op.x.rank == 3
    is_conv_transpose_2d = op.x.rank == 4
    is_conv_transpose_3d = op.x.rank == 5

    if is_conv_transpose_1d:
        x_name = op.name + "_expand_dim"
        out_name = op.name + "_expanded"
        builder.add_expand_dims(
            name=x_name, input_name=op.x.name, output_name=x_name, axes=[-2]
        )

    # Input names to be used
    input_names = [x_name]

    # Kernel shape: [C_in, C_out, D, H, W]
    weight = op.weight.val
    kernel_channels = weight.shape[0]
    output_channels = weight.shape[1] * op.groups.val

    if is_conv_transpose_1d:
        weight = _np.expand_dims(weight, -2)

    # pyMIL Deconvolution format: [C_in, C_out / groups, spatial_dims]
    # NN DeConvolution3D expects weights to have shape (C_out / groups, C_in, spatial_dims)
    # NN DeConvolution2D/1D expects (spatial_dims, C_in, C_out/groups)
    if is_conv_transpose_3d:
        weight = _np.transpose(weight, [1, 0, 2, 3, 4])
    else:
        weight = _np.transpose(weight, [2, 3, 0, 1])

    # Adjust for Deconv1D case
    # CoreML maps Deconv1D into Deconv2D
    # Hence, adjust width dimension attributes by setting to 1 for 1D case
    rank_factor = 1 if is_conv_transpose_1d else 2

    strides = op.strides.val.tolist()
    dilations = op.dilations.val.tolist()

    output_spatial_dims = list(op.outputs[0].shape[2:])
    if is_conv_transpose_1d:
        dilations = dilations[:-1] + [1] + dilations[-1:]
        strides = strides[:-1] + [1] + strides[-1:]
        # Must be at least 2D
        output_spatial_dims = output_spatial_dims[:-1] + [1] + output_spatial_dims[-1:]

    if any_symbolic(output_spatial_dims):
        output_spatial_dims = None

    # padding
    padding_mode = op.pad_type.val
    pad = {}
    if padding_mode == "custom":
        if is_conv_transpose_1d:
            padding_mode = "valid"
            pad["padding_top"] = 0
            pad["padding_bottom"] = 0
            pad["padding_left"] = op.pad.val[0]  # Left
            pad["padding_right"] = op.pad.val[1]  # Right
        elif is_conv_transpose_2d:
            padding_mode = "valid"
            pad["padding_top"] = op.pad.val[0]  # Top
            pad["padding_bottom"] = op.pad.val[1]  # Bottom
            pad["padding_left"] = op.pad.val[2]  # Left
            pad["padding_right"] = op.pad.val[3]  # Right
        else:
            pad["padding_front"] = op.pad.val[0]  # Front
            pad["padding_back"] = op.pad.val[1]  # Back
            pad["padding_top"] = op.pad.val[2]  # Top
            pad["padding_bottom"] = op.pad.val[3]  # Bottom
            pad["padding_left"] = op.pad.val[4]  # Left
            pad["padding_right"] = op.pad.val[5]  # Right

    groups = op.groups.val
    has_bias = op.bias is not None

    if is_conv_transpose_3d:
        builder.add_convolution3d(
            name=op.name,
            input_channels=kernel_channels,
            output_channels=output_channels,
            depth=weight.shape[-3],
            height=weight.shape[-2],
            width=weight.shape[-1],
            W=weight,
            b=op.bias.val if has_bias else None,
            has_bias=has_bias,
            groups=groups,
            stride_depth=strides[0],
            stride_height=strides[1],
            stride_width=strides[2],
            dilation_depth=dilations[0],
            dilation_height=dilations[1],
            dilation_width=dilations[2],
            padding_mode=padding_mode,
            is_deconv=True,
            output_shape=output_spatial_dims,
            input_name=input_names,
            output_name=out_name,
            **pad
        )
    else:
        builder.add_convolution(
            name=out_name,
            kernel_channels=kernel_channels,
            output_channels=output_channels,
            height=weight.shape[0],
            width=weight.shape[1],
            stride_height=strides[0],
            stride_width=strides[1],
            border_mode=padding_mode,
            groups=groups,
            W=weight,
            b=op.bias.val if has_bias else None,
            has_bias=has_bias,
            is_deconv=True,
            output_shape=output_spatial_dims,
            input_name=input_names,
            output_name=out_name,
            dilation_factors=dilations,
            **pad
        )

        # Squeeze added `Height` dimension for 1d case
        if is_conv_transpose_1d:
            builder.add_squeeze(
                name=op.name,
                input_name=out_name,
                output_name=op.outputs[0].name,
                axes=[-2],
            )


@register_mil_to_nn_mapping
def range_1d(const_context, builder, op):
    if op.start.val is not None and op.step.val is not None:
        inputs = [op.end]
    elif op.start.val is None and op.step.val is not None:
        inputs = [op.end, op.start]
    elif op.start.val is not None and op.step.val is None:
        inputs = [op.end, op.start, op.step]
    else:
        inputs = [op.end, op.start, op.step]

    builder.add_range_dynamic(
        name=op.name,
        output_name=op.outputs[0].name,
        input_names=make_input(const_context, builder, inputs),
        start=op.start.val if op.start.val is not None else 0,
        step=op.step.val if op.step.val is not None else 1,
    )


@register_mil_to_nn_mapping
def one_hot(const_context, builder, op):
    if op.one_hot_vector_size.val is not None:
        inputs = [op.indices]
    else:
        inputs = [op.indices, op.one_hot_vector_size]

    builder.add_one_hot(
        name=op.name,
        input_names=make_input(const_context, builder, inputs),
        output_name=op.outputs[0].name,
        one_hot_vector_size=op.one_hot_vector_size.val,
        axis=op.axis.val,
        on_value=op.on_value.val,
        off_value=op.off_value.val,
    )


@register_mil_to_nn_mapping
def non_maximum_suppression(const_context, builder, op):
    builder.add_nms(
        name=op.name,
        input_names=make_input(const_context, builder, [op.boxes, op.scores]),
        output_names=[op.outputs[i].name for i in range(4)],
        iou_threshold=op.iou_threshold.val,
        score_threshold=op.score_threshold.val,
        max_boxes=op.max_boxes.val,
        per_class_suppression=op.per_class_suppression.val,
    )


@register_mil_to_nn_mapping
def flatten2d(const_context, builder, op):
    builder.add_flatten_to_2d(
        name=op.name,
        input_name=make_input(const_context, builder, op.x),
        output_name=op.outputs[0].name,
        axis=op.axis.val,
    )


@register_mil_to_nn_mapping
def shape(const_context, builder, op):
    builder.add_get_shape(
        name=op.name,
        input_name=make_input(const_context, builder, op.x),
        output_name=op.outputs[0].name,
    )


def add_upsample_nn(const_context, builder, op, scale_factor_h, scale_factor_w):
    if _np.abs(_np.round(scale_factor_h) - scale_factor_h) < 1e-4 and scale_factor_h >= 1 - 1e-4:
        scale_factor_h = int(scale_factor_h)
    else:
        raise NotImplementedError(
            "Unsupported float type 'scale_factor_height' ({scale_factor_h}) for neuralnetwork."
        )
    if _np.abs(_np.round(scale_factor_w) - scale_factor_w) < 1e-4 and scale_factor_w >= 1 - 1e-4:
        scale_factor_w = int(scale_factor_w)
    else:
        raise NotImplementedError(
            "Unsupported float type 'scale_factor_width' ({scale_factor_w}) for neuralnetwork."
        )

    builder.add_upsample(
        name=op.name,
        scaling_factor_h=scale_factor_h,
        scaling_factor_w=scale_factor_w,
        input_name=make_input(const_context, builder, op.x),
        output_name=op.outputs[0].name,
        mode="NN",
    )


@register_mil_to_nn_mapping
def resize_nearest_neighbor(const_context, builder, op):
    Hout, Wout = op.target_size_height.val, op.target_size_width.val
    x_shape = op.x.shape
    Hin, Win = x_shape[-2], x_shape[-1]

    scale_factor_h = Hout / Hin if Hout % Hin == 0 else (Hout + 1e-4) / Hin
    scale_factor_w = Wout / Win if Wout % Win == 0 else (Wout + 1e-4) / Win

    add_upsample_nn(const_context, builder, op, scale_factor_h, scale_factor_w)


@register_mil_to_nn_mapping
def upsample_nearest_neighbor(const_context, builder, op):
    scale_factor_h = op.scale_factor_height.val
    scale_factor_w = op.scale_factor_width.val

    add_upsample_nn(const_context, builder, op, scale_factor_h, scale_factor_w)


@register_mil_to_nn_mapping
def upsample_bilinear(const_context, builder, op):
    builder.add_upsample(
        name=op.name,
        scaling_factor_h=op.scale_factor_height.val,
        scaling_factor_w=op.scale_factor_width.val,
        input_name=make_input(const_context, builder, op.x),
        output_name=op.outputs[0].name,
        mode="BILINEAR",
        linear_upsample_mode="ALIGN_CORNERS_TRUE" if op.align_corners.val else "ALIGN_CORNERS_FALSE",
    )


@register_mil_to_nn_mapping
def resize_bilinear(const_context, builder, op):
    grid_sampling_mode_map = {
        "STRICT_ALIGN_CORNERS": "STRICT_ALIGN_ENDPOINTS_MODE",
        "ALIGN_CORNERS": "ALIGN_ENDPOINTS_MODE",
        "DEFAULT": "UPSAMPLE_MODE",
        "OFFSET_CORNERS": "ROI_ALIGN_MODE"
    }

    if op.sampling_mode.val not in grid_sampling_mode_map:
        raise NotImplementedError(
            "Unsupported 'sampling_mode' ('{op.sampling_mode.val}') in neuralnetwork backend"
        )

    builder.add_resize_bilinear(
        name=op.name,
        input_name=make_input(const_context, builder, op.x),
        output_name=op.outputs[0].name,
        target_height=op.target_size_height.val,
        target_width=op.target_size_width.val,
        mode=grid_sampling_mode_map[op.sampling_mode.val],
    )


@register_mil_to_nn_mapping
def cond(const_context, builder, op):
    true_block = op.blocks[0]
    false_block = op.blocks[1]

    branch_layer = builder.add_branch(
        name=op.name, input_name=make_input(const_context, builder, op.pred),
    )
    true_builder = neural_network.NeuralNetworkBuilder(
        nn_spec=branch_layer.branch.ifBranch,
        disable_rank5_shape_mapping=True,
        use_float_arraytype=True,
    )
    convert_ops(const_context, true_builder, true_block.operations, true_block.outputs)

    # Copy block output to cond op output.
    for block_out, op_out in zip(true_block.outputs, op.outputs):
        true_builder.add_copy(
            name=block_out.name + "_ret_copy",
            # No need to make_input for block_out which is guaranteed
            # to be a node
            input_name=block_out.name,
            output_name=op_out.name,
        )

    false_builder = neural_network.NeuralNetworkBuilder(
        nn_spec=branch_layer.branch.elseBranch,
        disable_rank5_shape_mapping=True,
        use_float_arraytype=True,
    )
    convert_ops(
        const_context, false_builder, false_block.operations, false_block.outputs
    )

    for block_out, op_out in zip(false_block.outputs, op.outputs):
        false_builder.add_copy(
            name=block_out.name + "_ret_copy",
            input_name=block_out.name,
            output_name=op_out.name,
        )


@register_mil_to_nn_mapping
def while_loop(const_context, builder, op):
    cond_block = op.blocks[0]
    body_block = op.blocks[1]

    # Assume that all loop vars aren't loop invariant (invariant loop vars
    # should've be optimized away in graph passes).
    for v_in, vx_in in zip(op.loop_vars, cond_block.inputs):
        assert v_in.name != vx_in.name, "Loop invariant detected in {}".format(op)
        builder.add_copy(
            name=vx_in.name + "_input_copy",
            input_name=make_input(const_context, builder, v_in),
            output_name=vx_in.name,
        )

    loop_layer = builder.add_loop(
        name=op.name,
        # max_iterations=0 to use condition network.
        max_iterations=0,
    )

    # Construct while_loop condition
    cond_builder = neural_network.NeuralNetworkBuilder(
        nn_spec=loop_layer.loop.conditionNetwork,
        disable_rank5_shape_mapping=True,
        use_float_arraytype=True,
    )
    cond_builder.rank_dict = {k.name: builder.rank_dict[k.name] for k in cond_block.inputs}
    convert_ops(
        const_context,
        cond_builder,
        cond_block.operations,
        cond_block.outputs,
    )

    loop_layer.loop.conditionVar = cond_block.outputs[0].name

    # while_loop body produces loop_vars
    body_builder = neural_network.NeuralNetworkBuilder(
        nn_spec=loop_layer.loop.bodyNetwork,
        disable_rank5_shape_mapping=True,
        use_float_arraytype=True,
    )
    body_builder.rank_dict = {k.name: builder.rank_dict[k.name] for k in body_block.inputs}
    convert_ops(
        const_context,
        body_builder,
        body_block.operations,
        body_block.outputs,
    )

    # Also assume all outputs are different from loop inputs (i.e., no loop
    # invariant.)
    #for vx_in, vx_out in zip(block.inputs, block.outputs[1:]):
    for vx_in, vx_out in zip(body_block.inputs, body_block.outputs):
        if vx_in.name == vx_out.name:
            msg = "Loop invariant var {} detected in block {}"
            _logging.warning(msg.format(vx_in.name, body_block.name))
            continue
        body_builder.add_copy(
            name=vx_in.name + "_ret_copy",
            input_name=make_input(const_context, builder, vx_out),
            output_name=vx_in.name,
        )


@register_mil_to_nn_mapping
def identity(const_context, builder, op):
    builder.add_copy(
        name=op.name,
        input_name=make_input(const_context, builder, op.x),
        output_name=op.outputs[0].name,
    )


@register_mil_to_nn_mapping
def concat(const_context, builder, op):
    # filter out input tensor with 0 size
    values = []
    for v in op.values:
        if len(v.shape) > 0 and v.shape[op.axis.val] == 0:
            continue
        values.append(v)

    if len(values) == 0:
        raise NotImplementedError('0 size tensor unsupported.')

    if len(values) >= 2:
        rank = values[0].rank
        if op.interleave.val:
            builder.add_concat_nd(
                    name=op.name,
                    input_names=make_input(const_context, builder, values),
                    output_name=op.outputs[0].name,
                    axis=op.axis.val,
                    interleave=True)
        elif rank >= 4 and (op.axis.val == -3 or op.axis.val > 0 and op.axis.val == rank - 3):
            builder.add_elementwise(
                name=op.name,
                input_names=make_input(const_context, builder, values),
                output_name=op.outputs[0].name,
                mode="CONCAT",
            )
        else:
            builder.add_concat_nd(
                    name=op.name,
                    input_names=make_input(const_context, builder, values),
                    output_name=op.outputs[0].name,
                    axis=op.axis.val)
    else:
        builder.add_copy(
                name=op.name,
                input_name=make_input(const_context, builder, values[0]),
                output_name=op.outputs[0].name)


@register_mil_to_nn_mapping
def stack(const_context, builder, op):
    builder.add_stack(
        name=op.name,
        input_names=make_input(const_context, builder, op.values),
        output_name=op.outputs[0].name,
        axis=op.axis.val,
    )


@register_mil_to_nn_mapping
def split(const_context, builder, op):
    split_sizes = None
    if op.split_sizes is not None:
        if op.split_sizes.val is None:
            raise ValueError('Non-const split_sizes unsupported in NN')
        split_sizes = op.split_sizes.val.tolist()

    split = op.sizes
    split = [size for size in split if size != 0]
    has_equal_splits = all([size == split[0] for size in split])
    num_splits = len(split)
    output_names = [op.outputs[i].name for i in range(len(op.sizes)) if op.sizes[i] != 0]

    if has_equal_splits:
        builder.add_split_nd(
                name=op.name,
                input_name=make_input(const_context, builder, op.x),
                output_names=output_names,
                axis=op.axis.val,
                num_splits=num_splits)
    else:
        builder.add_split_nd(
                name=op.name,
                input_name=make_input(const_context, builder, op.x),
                output_names=output_names,
                axis=op.axis.val,
                split_sizes=list(split))


@register_mil_to_nn_mapping
def argsort(const_context, builder, op):
    axis = op.x.rank + op.axis.val if op.axis.val < 0 else op.axis.val
    builder.add_argsort(
        name=op.name,
        input_name=make_input(const_context, builder, op.x),
        output_name=op.outputs[0].name,
        axis=axis,
        descending=(not op.ascending.val),
    )


@register_mil_to_nn_mapping
def pixel_shuffle(const_context, builder, op):
    builder.add_reorganize_data(
        name=op.name,
        input_name=make_input(const_context, builder, op.x),
        output_name=op.outputs[0].name,
        mode="PIXEL_SHUFFLE",
        block_size=op.upscale_factor.val,
    )


@register_mil_to_nn_mapping
def sliding_windows(const_context, builder, op):
    builder.add_sliding_windows(
        name=op.name,
        input_name=make_input(const_context, builder, op.x),
        output_name=op.outputs[0].name,
        axis=op.axis.val,
        window_size=op.size.val,
        step=op.stride.val,
    )


@register_mil_to_nn_mapping
def crop(const_context, builder, op):
    builder.add_crop(
        name=op.name,
        input_names=[op.x.name],
        output_name=op.outputs[0].name,
        offset=0,
        left=op.crop_width.val[0],
        right=op.crop_width.val[1],
        top=op.crop_height.val[0],
        bottom=op.crop_height.val[1],
    )


@register_mil_to_nn_mapping
def crop_resize(const_context, builder, op):
    grid_sampling_mode_map = {
        "STRICT_ALIGN_CORNERS": "STRICT_ALIGN_ENDPOINTS_MODE",
        "ALIGN_CORNERS": "ALIGN_ENDPOINTS_MODE",
        "DEFAULT": "UPSAMPLE_MODE",
        "OFFSET_CORNERS": "ROI_ALIGN_MODE",
    }

    if op.sampling_mode.val not in grid_sampling_mode_map:
        raise NotImplementedError(
            "Unsupported 'sampling_mode' ('{}') in neuralnetwork backend".format(
                op.sampling_mode.val
            )
        )

    mode = grid_sampling_mode_map[op.sampling_mode.val]

    input_expanded = op.name + "_x_expand"
    builder.add_expand_dims(
        name=input_expanded,
        input_name=make_input(const_context, builder, op.x),
        output_name=input_expanded,
        axes=[0],
    )
    builder.add_crop_resize(
        name=op.name,
        input_names=make_input(const_context, builder, [input_expanded, op.roi]),
        output_name=op.outputs[0].name,
        target_height=op.target_height.val,
        target_width=op.target_width.val,
        mode=mode,
        normalized_roi=op.normalized_coordinates.val,
        box_indices_mode=op.box_coordinate_mode.val,
        spatial_scale=op.spatial_scale.val,
    )


@register_mil_to_nn_mapping
def custom_op(const_context, builder, op):
    class_name = op.bindings.get("class_name", op.name)
    input_order = op.bindings.get("input_order", [])
    parameters = op.bindings.get("parameters", [])
    weights = op.bindings.get("weights", [])
    description = op.bindings.get("description", "")

    if len(input_order) == 0:
        raise ValueError("Inputs not provided for Custom Layer: {}".format(op.name))

    # Get input names
    inputs = [op.inputs[_name] for _name in input_order]

    # Get output names
    output_names = [_output.name for _output in op.outputs]

    # Load custom params
    params = NeuralNetwork_pb2.CustomLayerParams()
    params.className = class_name
    params.description = description

    # Load parameters
    for _param in parameters:
        param = op.inputs[_param]
        param_val = param.val
        if types.is_bool(param.dtype):
            params.parameters[_param].boolValue = param_val
        elif types.is_int(param.dtype):
            params.parameters[_param].intValue = param_val
        elif types.is_float(param.dtype):
            params.parameters[_param].doubleValue = param_val
        elif types.is_str(param.dtype):
            params.parameters[_param].stringValue = param_val
        else:
            raise ValueError(
                "Unknown parameter type for custom layer- "
                "Op: {}, Parameter: {}, Type: {}".format(op.name, _param, param.dtype)
            )

    # Load weights
    for _weight in weights:
        wt = params.weights.add()
        wt.floatValue.extend(map(float, _weight))

    # Add a custom layer
    builder.add_custom(
        name=op.name,
        input_names=make_input(const_context, builder, inputs),
        output_names=output_names,
        custom_proto_spec=params,
    )


@register_mil_to_nn_mapping
def make_list(const_context, builder, op):
    # Set a initial size
    size = op.init_length.val

    # set the dynamic dimensions to 1 for initialization
    # Ex: op.elem_shape = [i0, 128] will result in [1, 128]
    elem_shape = [1 if isinstance(dim_var.val, str) else
        dim_var.val for dim_var in op.elem_shape]

    if size is not None:
        array_size = size if size > 0 else 1
        array_shape = [array_size] + elem_shape
        add_const(
            const_context,
            builder,
            op.outputs[0].name,
            val=_np.zeros(array_shape, dtype="float"),
        )
    else:
        if len(elem_shape) > 0:
            node_es_name = op.name + "_element_shape"
            add_const(
                const_context,
                builder,
                node_es_name,
                val=_np.array(elem_shape, dtype="float"),
            )

            # Concatenate list length of the input, should be a constant vector of size 1) with element shape
            node_arr_shape_name = op.name + "_arr_shape"
            layer = builder.add_concat_nd(
                name=node_arr_shape_name,
                input_names=[op.init_length.name, node_es_name],
                output_name=node_arr_shape_name,
                axis=0,
            )
        else:
            raise ValueError("elem_shape should have length > 0.")

        builder.add_fill_dynamic(
            name=op.name, input_name=node_arr_shape_name, output_name=op.outputs[0].name
        )


def _realloc_list(const_context, builder, ls_var, index_var, value_var, mode):
    # we do two things in this helper function
    # (1)
    # check if we need to re-initialize the tensorarray:
    # it happens when the elem_shape is runtime determined and the runtime shape is not equal to
    # the default shape. Ex: elem_shape is = [i0, 10] (initilized with [1, 10]) and at the runtime we get [2, 10].

    # (2)
    # If index_var >= len(ls_var), reallocate the array and copy over existing
    # contents

    # index_var: str or Var
    # ls_var: Var

    # check if elem_shape is runtime-determined
    elem_shape = tuple(value_var.shape)
    has_dynamic_shape = any([is_symbolic(i) for i in elem_shape])

    # get the fill shape of the tensor array
    # [length, elem_dim1, elem_dim2, ...]
    full_shape_name = ls_var.name + "_full_shape"
    builder.add_get_shape(
        name=full_shape_name,
        input_name=ls_var.name,  # no need to make_input
        output_name=full_shape_name,
    )

    # slice shape [length, elem_dim1, elem_dim2, ...] to get current length
    curr_len_name = ls_var.name + "_length"
    builder.add_slice_static(
        name=curr_len_name,
        input_name=full_shape_name,
        output_name=curr_len_name,
        begin_ids=[0],
        end_ids=[1],
        begin_masks=[False],
        end_masks=[False],
        strides=[1],
    )

    value_elem_shape_name = ls_var.name + '_value_elem_shape'
    if has_dynamic_shape:
        # get elem_shape from value if it is runtime-determined
        # this is similar to what the backfill_make_list_elem_type tf graph pass does.
        # if mode == "list_write", elem_shape equal to value.shape,
        # if mode == "list_scatter", elem_shape equal to value.shape[1:]
        if mode == "list_write":
            builder.add_get_shape(
                name=value_elem_shape_name,
                input_name=make_input(const_context, builder, value_var),
                output_name=value_elem_shape_name,
            )
        elif mode == "list_scatter":
            raw_value_elem_shape_name = ls_var.name + '_raw_value_elem_shape'
            builder.add_get_shape(
                name=raw_value_elem_shape_name,
                input_name=make_input(const_context, builder, value_var),
                output_name=raw_value_elem_shape_name,
            )

            builder.add_slice_static(
                name=value_elem_shape_name,
                input_name=raw_value_elem_shape_name,
                output_name=value_elem_shape_name,
                begin_ids=[1],
                end_ids=[-1],
                begin_masks=[False],
                end_masks=[True],
                strides=[1],
            )
    else:
        add_const(const_context, builder, value_elem_shape_name, _np.array(elem_shape))


    # if elem_shape is runtime-determined, check if we need to re-initialize the array

    if has_dynamic_shape:
        # slice shape [length, elem_dim1, elem_dim2, ...] to get list elem_shape
        curr_elem_shape_name = ls_var.name + "_ls_elem_shape"
        builder.add_slice_static(
            name=curr_elem_shape_name,
            input_name=full_shape_name,
            output_name=curr_elem_shape_name,
            begin_ids=[1],
            end_ids=[-1],
            begin_masks=[False],
            end_masks=[True],
            strides=[1],
        )

        # test if the runtime elem_shape from the list and value are equal
        not_equal_name = ls_var.name + '_elem_shape_not_equal'
        builder.add_not_equal(
            name=not_equal_name,
            input_names=[curr_elem_shape_name, value_elem_shape_name],
            output_name=not_equal_name,
        )

        reduce_any_name = ls_var.name + '_reduce_any'
        builder.add_reduce_sum(
            name=reduce_any_name,
            input_name=not_equal_name,
            output_name=reduce_any_name,
            axes=[0],
            keepdims=False,
            reduce_all=True,
        )

        # if the two elem_shape are different, then re initialize the list with elem_shape from the value
        re_initialize_condition_name = ls_var.name + "_condition_re_initialize"
        layer = builder.add_branch(name=re_initialize_condition_name, input_name=reduce_any_name)
        true_builder = neural_network.NeuralNetworkBuilder(
            nn_spec=layer.branch.ifBranch,
            disable_rank5_shape_mapping=True,
            use_float_arraytype=True,
        )

        re_initialize_shape_name = ls_var.name + "_re_initialize_shape"
        true_builder.add_concat_nd(
            name=re_initialize_shape_name,
            input_names=[curr_len_name, value_elem_shape_name],
            output_name=re_initialize_shape_name,
            axis=0,
        )

        re_initialize_name = ls_var.name + "_re_initialize"
        true_builder.add_fill_dynamic(
            name=re_initialize_name,
            input_name=re_initialize_shape_name,
            output_name=re_initialize_name,
            value=0.0,
        )

        true_builder.add_copy(
            name=ls_var.name + "_re_initialize_assign",
            input_name=re_initialize_name,
            output_name=ls_var.name
        )

    # after re-initialize the list, we now check if we need to reallocate the list
    # check if the index > curr_length
    is_growing_name = ls_var.name + "_is_growing"
    builder.add_greater_than(
        name=is_growing_name,
        input_names=make_input(const_context, builder, [index_var, curr_len_name]),
        output_name=is_growing_name,
        use_greater_than_equal=True,
    )

    condition_name = ls_var.name + "_condition"
    layer = builder.add_branch(name=condition_name, input_name=is_growing_name)

    true_builder = neural_network.NeuralNetworkBuilder(
        nn_spec=layer.branch.ifBranch,
        disable_rank5_shape_mapping=True,
        use_float_arraytype=True,
    )

    # alloc_length_name0 = index - list_length
    alloc_length_name0 = ls_var.name + "_extra_length0"
    true_builder.add_subtract_broadcastable(
        name=alloc_length_name0,
        input_names=make_input(const_context, builder, [index_var, curr_len_name]),
        output_name=alloc_length_name0,
    )

    # alloc_length_name1 = index - list_length + 1
    alloc_length_name1 = ls_var.name + "_extra_length1"
    true_builder.add_elementwise(
        name=alloc_length_name1,
        input_names=[alloc_length_name0],
        mode="ADD",
        output_name=alloc_length_name1,
        alpha=1,
    )

    # alloc_shape_name = [alloc_length] + elem_shape
    alloc_shape_name = ls_var.name + "_alloc_shape"
    true_builder.add_concat_nd(
        name=alloc_shape_name,
        input_names=[alloc_length_name1, value_elem_shape_name],
        output_name=alloc_shape_name,
        axis=0,
    )

    # new_alloc_name is np.zeros([alloc_length] + elem_shape)
    new_alloc_name = ls_var.name + "_alloc"
    true_builder.add_fill_dynamic(
        name=new_alloc_name,
        input_name=alloc_shape_name,
        output_name=new_alloc_name,
        value=0.0,
    )

    # new_list_name is np.concat([old_list, new_alloc])
    new_list_name = ls_var.name + "_new"
    true_builder.add_concat_nd(
        name=new_list_name,
        input_names=[ls_var.name, new_alloc_name],
        output_name=new_list_name,
        axis=0,
    )

    # Copy new_list_name to ls_var.name
    true_builder.add_copy(
        name=ls_var.name + "_assign", input_name=new_list_name, output_name=ls_var.name
    )


@register_mil_to_nn_mapping
def list_write(const_context, builder, op):
    _realloc_list(const_context, builder, op.ls, op.index, op.value, "list_write")

    # expanded_value_name is [1, op.value]
    expanded_value_name = op.ls.name + '_' + op.value.name + "_expanded"
    builder.add_expand_dims(
        name=expanded_value_name,
        input_name=make_input(const_context, builder, op.value),
        output_name=expanded_value_name,
        axes=[0],
    )

    builder.add_scatter(
        name=op.name,
        input_names=make_input(
            const_context, builder, [op.ls, op.index, expanded_value_name]
        ),
        output_name=op.outputs[0].name,
    )


@register_mil_to_nn_mapping
def list_gather(const_context, builder, op):
    builder.add_gather(
        name=op.name,
        input_names=make_input(const_context, builder, [op.ls, op.indices]),
        output_name=op.outputs[0].name,
        axis=0,
    )


@register_mil_to_nn_mapping
def list_scatter(const_context, builder, op):
    max_idx_name = op.indices.name + "_max"
    builder.add_reduce_max(
        name=max_idx_name,
        axes=[0],
        keepdims=False,
        input_name=make_input(const_context, builder, op.indices),
        output_name=max_idx_name,
    )
    _realloc_list(const_context, builder, op.ls, max_idx_name, op.value, "list_scatter")
    builder.add_scatter(
        name=op.name,
        input_names=make_input(const_context, builder, [op.ls, op.indices, op.value]),
        output_name=op.outputs[0].name,
    )


@register_mil_to_nn_mapping
def list_read(const_context, builder, op):
    # gathered_name has shape [1] + elem_shape
    gathered_name = op.name + "_gathered"
    builder.add_gather(
        name=op.name,
        input_names=make_input(const_context, builder, [op.ls, op.index]),
        output_name=gathered_name,
        axis=0,
    )

    # squeezed_name has shape elem_shape
    squeezed_name = op.name + "_squeezed"
    builder.add_squeeze(
        name=squeezed_name,
        input_name=gathered_name,
        output_name=op.outputs[0].name,
        axes=[0],
    )


@register_mil_to_nn_mapping
def list_length(const_context, builder, op):
    # list_shape_name == [list_length] + elem_shape
    list_shape_name = op.ls.name + "_shape"
    builder.add_get_shape(
        name=list_shape_name,
        input_name=make_input(const_context, builder, op.ls),
        output_name=list_shape_name,
    )

    # slice to get list_length
    builder.add_slice_static(
        name=op.name,
        input_name=list_shape_name,
        output_name=op.outputs[0].name,
        begin_ids=[0],
        end_ids=[1],
        begin_masks=[False],
        end_masks=[False],
        strides=[1],
    )

@register_mil_to_nn_mapping
def _const_symbolic(const_context, builder, op):
    # do nothing
    pass

