#  Copyright (c) 2023, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import copy
from typing import List, Optional, Tuple

import numpy as np

from coremltools import _logger as logger
from coremltools.converters.mil.mil import Block
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import Operation, Program, types
from coremltools.converters.mil.mil.passes.graph_pass import AbstractGraphPass
from coremltools.converters.mil.mil.passes.helper import (
    _check_child_op_type,
    _check_no_output_connection,
    block_context_manager,
)
from coremltools.converters.mil.mil.passes.pass_registry import register_pass
from coremltools.converters.mil.mil.types.symbolic import any_symbolic


@register_pass(namespace="common")
class add_conv_transpose_output_shape(AbstractGraphPass):
    """
    The ``conv_transpose`` input ``output_shape`` is an optional input.
    Since we can infer the output shape from ``type_inference``, we add
    ``output_shape`` input whenever it is known to be constant at
    compile time. For example:

    .. code-block::

        Given:
          %1: (1, 5, 39, fp32) = conv_transpose(...) # no output_shape input.

        Result:
          %2: (3, i32) = const(val=[1,5,39])
          %3: (1, 5, 39, fp32) = conv_transpose(..., output_shape=%2)
    """

    def apply(self, prog):
        for f in prog.functions.values():
            self._handle_block(f)

    @staticmethod
    def _match_pattern(op):
        return (
            op.op_type == "conv_transpose"
            and op.output_shape is None
            and not any_symbolic(op.outputs[0].shape)
        )

    @block_context_manager
    def _handle_block(self, block):
        for op in list(block.operations):
            for b in op.blocks:
                self._handle_block(b)

            if not self._match_pattern(op):
                continue

            # matched pattern
            x = mb.conv_transpose(
                **op.inputs,
                output_shape=op.outputs[0].shape,
                name=op.name + "_has_output_shape",
                before_op=op,
            )
            op.enclosing_block.replace_uses_of_var_after_op(
                anchor_op=op, old_var=op.outputs[0], new_var=x
            )
            block.remove_ops([op])


@register_pass(namespace="common")
class compose_conv1d(AbstractGraphPass):
    """
    In `TensorFlow <https://github.com/tensorflow/tensorflow/blob/r1.15/tensorflow/python/ops/nn_ops.py#L1657>`_,
    ``tf.keras.layers.Conv1D`` is a composite op:

    .. code-block::

        expand a dummy dim -> Conv2D -> squeeze the dummy dim

    In `PyTorch <https://github.com/pytorch/pytorch/blob/release/1.13/aten/src/ATen/native/Convolution.cpp#L1087>`_,
    this is also true for some backends (``mkldnn`` and ``xpu``).

    This decomposition wrecks the coremltools ``conv1d`` graph passes,
    so we should recompose the fragments back to MIL ``conv``, which natively supports ``conv1d``:

    .. code-block::

        Pattern 1:
            Given:
                %2 = expand_dims(%1, axes=-2) or expand_dims(%1, axes=2), %1.rank = 3
                %3 = conv(%2)
                %4 = squeeze(%3, axes=-2) or squeeze(%3, axes=2)
                ...

            Result:
                %4 = conv(%1)
                ...

        Pattern 2 (TensorFlow channel_last):
            Given:
                %2 = expand_dims(%1, axes=-3) or expand_dims(%1, axes=1), %1.rank = 3
                %3 = transpose(%2, perm=(0, 3, 1, 2))
                %4 = conv(%3)
                %5 = transpose(%4, perm=(0, 2, 3, 1))
                %6 = squeeze(%5, axes=-3) or squeeze(%5, axes=1)
                ...

            Result:
                %3 = transpose(%1, perm=(0, 2, 1))
                %4 = conv(%3)
                %6 = transpose(%4, perm=(0, 2, 1))
                ...
    """

    def apply(self, prog):
        for f in prog.functions.values():
            self._compose_conv1d_block(f)

    @block_context_manager
    def _compose_conv1d_block(self, block: Block):
        def help_compose_conv1d_block(block: Block) -> bool:
            fusion_occurred = False
            for op in list(block.operations):
                if op.enclosing_block is None:
                    continue

                for b in op.blocks:
                    self._compose_conv1d_block(b)

                # must start with expanding a 3-D tensor,
                # who has batch, channel, length dimensions
                if op.op_type != "expand_dims" or op.x.rank != 3:
                    continue

                # try pattern `expand_dim` -> `conv2d` -> `squeeze`
                if self._try_match_and_transform_pattern(op, block):
                    # has to break as the downstream iterator is affected
                    return True

                # try pattern `expand_dim` -> `transpose` -> `conv2d` -> `transpose` -> `squeeze`
                if self._try_match_and_transform_pattern_channel_last(op, block):
                    fusion_occurred = True

            return fusion_occurred

        block_changed = True
        while block_changed:
            block_changed = help_compose_conv1d_block(block)

    def _try_match_and_transform_pattern(self, expand_op: Operation, block: Block) -> bool:
        """
        identify the pattern: `expand_dim` -> `conv2d` -> `squeeze`
        """
        # abort composition if dummy dimension is not added as height
        if expand_op.axes.rank != 1 or expand_op.axes.val[0] not in (-2, 2):
            return False

        # `expand_dims` -> `conv`
        if not _check_child_op_type(expand_op, "conv"):
            return False
        conv_op = expand_op.outputs[0].child_ops[0]

        # `conv` -> `squeeze`
        if not _check_child_op_type(conv_op, "squeeze"):
            return False
        squeeze_op = conv_op.outputs[0].child_ops[0]

        # Abort composition if not squeezing the dummy height (the extended dim_size=1 dimension)
        if squeeze_op.axes.rank != 1 or squeeze_op.axes.val[0] not in (-2, 2):
            return False
        elif squeeze_op.x.shape[squeeze_op.axes.val[0]] != 1:
            return False

        # everything looks good
        return self._try_apply_transform(expand_op, conv_op, squeeze_op, block)

    def _try_match_and_transform_pattern_channel_last(
        self, expand_op: Operation, block: Block
    ) -> bool:
        """
        identify the pattern: `expand_dim` -> `transpose` -> `conv2d` -> `transpose` -> `squeeze`
        """
        # abort composition if dummy dimension is not added as height
        if expand_op.axes.rank != 1 or expand_op.axes.val[0] not in (-3, 1):
            return False

        # `expand_dims` -> `transpose`
        if not _check_child_op_type(expand_op, "transpose"):
            return False
        transpose1_op = expand_op.outputs[0].child_ops[0]

        # abort composition if permutation is not (0, 3, 1, 2)
        perm1 = transpose1_op.perm.val.copy()
        perm1[np.where(perm1 < 0)] += 4
        if np.any(perm1 != (0, 3, 1, 2)):
            return False

        # `transpose` -> `conv`
        if not _check_child_op_type(transpose1_op, "conv"):
            return False
        conv_op = transpose1_op.outputs[0].child_ops[0]

        # `conv` -> `transpose`
        if not _check_child_op_type(conv_op, "transpose"):
            return False
        transpose2_op = conv_op.outputs[0].child_ops[0]

        # abort composition if permutation is not (0, 2, 3, 1)
        perm2 = transpose2_op.perm.val.copy()
        perm2[np.where(perm2 < 0)] += 4
        if np.any(perm2 != (0, 2, 3, 1)):
            return False

        # `transpose` -> `squeeze`
        if not _check_child_op_type(transpose2_op, "squeeze"):
            return False
        squeeze_op = transpose2_op.outputs[0].child_ops[0]

        # abort composition if not squeezing the dummy height
        if squeeze_op.axes.rank != 1 or squeeze_op.axes.val[0] not in (-3, 1):
            return False

        # everything looks good
        return self._try_apply_transform_channel_last(
            expand_op, transpose1_op, conv_op, transpose2_op, squeeze_op, block
        )

    @staticmethod
    def _try_apply_transform(
        expand_op: Operation, conv_op: Operation, squeeze_op: Operation, block: Block
    ) -> bool:
        ops_to_remove = [expand_op, conv_op, squeeze_op]
        if not _check_no_output_connection(block, ops_to_remove):
            return False

        # prepare `conv1d`
        conv_kwargs = {"name": squeeze_op.outputs[0].name, "before_op": conv_op}

        # inherit `x` from `expand_dim`
        conv_kwargs["x"] = expand_op.x

        # inherit `pad_type`, `groups`, `bias` from `conv2d`
        conv_kwargs["pad_type"] = conv_op.inputs["pad_type"].val
        conv_kwargs["groups"] = conv_op.inputs["groups"].val
        bias = conv_op.inputs.get("bias", None)
        if bias is not None:
            conv_kwargs["bias"] = bias

        # squeeze `weight`, `strides`, `pad`, `dilations` from `conv2d`
        conv_kwargs["weight"] = mb.squeeze(
            x=conv_op.inputs["weight"], axes=(-2,), before_op=conv_op
        )
        conv_kwargs["strides"] = (conv_op.inputs["strides"].val[-1],)
        conv_kwargs["pad"] = (conv_op.inputs["pad"].val[-2], conv_op.inputs["pad"].val[-1])
        conv_kwargs["dilations"] = (conv_op.inputs["dilations"].val[-1],)

        # compose `conv1d`
        out = mb.conv(**conv_kwargs)

        # try replacing `expand_dim` -> `conv2d` -> `squeeze` output
        # with the new `conv1d` output
        if squeeze_op.enclosing_block.try_replace_uses_of_var_after_op(
            anchor_op=squeeze_op, old_var=squeeze_op.outputs[0], new_var=out
        ):
            # remove `expand_dim` -> `conv2d` -> `squeeze`
            block.remove_ops(ops_to_remove)
            return True
        return False

    @staticmethod
    def _try_apply_transform_channel_last(
        expand_op: Operation,
        transpose1_op: Operation,
        conv_op: Operation,
        transpose2_op: Operation,
        squeeze_op: Operation,
        block: Block,
    ) -> bool:
        ops_to_remove = [expand_op, transpose1_op, conv_op, transpose2_op, squeeze_op]
        if not _check_no_output_connection(block, ops_to_remove):
            return False

        # create `transpose1`
        transpose1_out = mb.transpose(
            x=expand_op.x, perm=(0, 2, 1), name=transpose1_op.outputs[0].name, before_op=expand_op
        )

        # prepare `conv1d`
        conv_kwargs = {"name": conv_op.outputs[0].name, "x": transpose1_out, "before_op": conv_op}

        # inherit `pad_type`, `groups`, `bias` from `conv2d`
        conv_kwargs["pad_type"] = conv_op.inputs["pad_type"].val
        conv_kwargs["groups"] = conv_op.inputs["groups"].val
        bias = conv_op.inputs.get("bias", None)
        if bias is not None:
            conv_kwargs["bias"] = bias

        # squeeze `weight`, `strides`, `pad`, `dilations` from `conv2d`
        conv_kwargs["weight"] = mb.squeeze(
            x=conv_op.inputs["weight"], axes=(-2,), before_op=conv_op
        )
        conv_kwargs["strides"] = (conv_op.inputs["strides"].val[-1],)
        conv_kwargs["pad"] = (conv_op.inputs["pad"].val[-2], conv_op.inputs["pad"].val[-1])
        conv_kwargs["dilations"] = (conv_op.inputs["dilations"].val[-1],)

        # compose `conv1d`
        conv_out = mb.conv(**conv_kwargs)

        # create `transpose2`
        transpose2_out = mb.transpose(
            x=conv_out, perm=(0, 2, 1), name=squeeze_op.outputs[0].name, before_op=transpose2_op
        )

        # try replacing `expand_dim` -> `transpose` -> `conv2d` -> `transpose` -> `squeeze` output
        # with the new `transpose` -> `conv1d` -> `transpose` output
        if squeeze_op.enclosing_block.try_replace_uses_of_var_after_op(
            anchor_op=squeeze_op, old_var=squeeze_op.outputs[0], new_var=transpose2_out
        ):
            # remove `expand_dim` -> `transpose` -> `conv2d` -> `transpose` -> `squeeze`
            block.remove_ops(ops_to_remove)
            return True
        return False


@register_pass(namespace="common")
class fuse_conv_batchnorm(AbstractGraphPass):
    """
    Fuse the following ``batch_norm`` layer into ``conv`` and ``conv_transpose``.
    That is, convert ``conv + batch_norm`` to ``conv``, by modifying the weight and bias in the ``conv`` layer.

    .. code-block::

        Given:
            %2 = conv(%1)
            ...
            %3 = batch_norm(%2)
            ...

        Result:
            %3 = conv(%1)
            ...
    """

    def apply(self, prog):
        for f in prog.functions.values():
            block_changed = True
            while block_changed:
                block_changed = self._fuse_conv_batchnorm_block(f)

    @staticmethod
    def _try_to_transform(conv_op, bn_op):
        # get parameters from batch_norm layer
        gamma = bn_op.gamma.val
        beta = bn_op.beta.val
        mean = bn_op.mean.val
        variance = bn_op.variance.val
        epsilon = bn_op.epsilon.val

        # get weight, bias and groups from conv layer
        if conv_op.weight.val is None:
            return False
        conv_weight = conv_op.weight.val
        conv_bias = conv_op.bias
        groups = conv_op.groups.val

        # get type of the conv layer
        is_deconv = conv_op.op_type == "conv_transpose"
        # The deconv weight transpose axes is determined by the dimension of convolution.
        # Conv1d should be [1, 0, 2], Conv2d should be [1, 0, 2, 3], Conv3d should be [1, 0, 2, 3, 4]
        if not 3 <= len(conv_weight.shape) <= 5:
            raise AssertionError(
                f"Only supports Conv1/2/3d, which means weight's dimension should"
                f"between 3 and 5, but got weight with {len(conv_weight.shape)} "
                f"dimensions. "
            )
        deconv_weight_transpose_axes = [1, 0] + [axis for axis in range(2, len(conv_weight.shape))]

        # D_in denotes the spatial dimensions for conv kernel weight
        # for conv_transpose, conv_weight has shape [Cin, Cout / groups, *D_in]
        # for conv, conv_weight has shape [Cout, Cin / groups, *D_in]
        if is_deconv:
            Cout = conv_weight.shape[1] * groups
            Cin = conv_weight.shape[0]
        else:
            Cout = conv_weight.shape[0]
            Cin = conv_weight.shape[1] * groups

        # get the type of the conv weight
        conv_weight_type = conv_weight.dtype

        # create bias for conv if not exist
        if conv_bias is None:
            conv_bias = np.zeros(Cout)
        else:
            conv_bias = conv_bias.val

        if conv_bias is None:
            return False

        conv_bias = conv_bias.astype(conv_weight_type)

        # get the original shape of weight and bias
        origin_weight_shape = conv_weight.shape
        origin_bias_shape = conv_bias.shape

        # update the weight for conv layer
        new_conv_weight = []
        new_conv_bias = []

        if is_deconv:
            conv_weight = np.transpose(conv_weight, deconv_weight_transpose_axes)
            conv_weight = np.reshape(
                conv_weight, [Cout, Cin // groups] + list(conv_weight.shape[2:])
            )

        for i in range(Cout):
            # get batch norm parameters for each channel
            _gamma = gamma[i]
            _beta = beta[i]
            _mean = mean[i]
            _variance = variance[i]
            _scale = _gamma / np.sqrt(_variance + epsilon)

            # get conv weight and bias for each channel
            _conv_weight = conv_weight[i]
            _conv_bias = conv_bias[i]

            # update the conv weight and bias
            _conv_weight = _conv_weight * _scale
            _conv_bias = _scale * (_conv_bias - _mean) + _beta
            new_conv_weight.append(_conv_weight)
            new_conv_bias.append(_conv_bias)

        new_conv_weight = np.array(new_conv_weight).astype(conv_weight_type)
        new_conv_bias = np.array(new_conv_bias).astype(conv_weight_type)

        if is_deconv:
            new_conv_weight = np.reshape(
                new_conv_weight, [Cout // groups, Cin] + list(new_conv_weight.shape[2:])
            )
            new_conv_weight = np.transpose(new_conv_weight, deconv_weight_transpose_axes)

        # make sure the updated weight and bias have the same shape as the original ones
        if new_conv_weight.shape != origin_weight_shape:
            raise AssertionError(
                "conv weight should have the same shape before and after the fuse_"
                "conv_batchnorm pass. "
            )
        if new_conv_bias.shape != origin_bias_shape:
            raise AssertionError(
                "conv bias should have the same shape before and after the fuse_"
                "conv_batchnorm pass. "
            )

        # the new weight / bias should inherit the meta data from the old conv layer
        # TODO: this is currently a temporary solution, we should consider a more general approach.
        # the follow-up is tracked by: rdar://131637107
        new_conv_weight = mb.const(val=new_conv_weight, before_op=conv_op)
        new_conv_bias = mb.const(val=new_conv_bias, before_op=conv_op)

        if conv_op.weight.op.op_type == "const":
            block = conv_op.enclosing_block
            block._copy_metadata(conv_op.weight, new_conv_weight)
            block._copy_metadata(conv_op.weight, new_conv_bias)

        # create a new conv op with the new bias value, copying rest of the attributes
        out_name = bn_op.outputs[0].name
        conv_kargs = {
            "weight": new_conv_weight,
            "bias": new_conv_bias,
            "name": out_name,
            "before_op": conv_op,
        }

        for k, v in conv_op.inputs.items():
            if k in ["weight", "bias"]:
                continue
            conv_kargs[k] = v

        if is_deconv:
            x = mb.conv_transpose(**conv_kargs)
        else:
            x = mb.conv(**conv_kargs)

        if bn_op.enclosing_block.try_replace_uses_of_var_after_op(
            anchor_op=bn_op,
            old_var=bn_op.outputs[0],
            new_var=x,
        ):
            bn_op.enclosing_block.remove_ops([conv_op, bn_op])
            return True
        return False

    @block_context_manager
    def _fuse_conv_batchnorm_block(self, block):
        def _match_pattern(op):
            if op.op_type == "conv" or op.op_type == "conv_transpose":
                # abort fusion if op output is also a block output
                if op.outputs[0] in op.enclosing_block.outputs:
                    return None
                # find batch_norm op
                child_ops = op.outputs[0].child_ops
                if len(child_ops) == 1:
                    bn_op_candidate = list(child_ops)[0]
                    if bn_op_candidate.op_type == "batch_norm":
                        return bn_op_candidate
            return None

        fusion_occurred = False
        for op in list(block.operations):
            if op.enclosing_block is None:
                continue

            for b in op.blocks:
                block_changed = True
                while block_changed:
                    block_changed = self._fuse_conv_batchnorm_block(b)
            if len(op.blocks) > 0:
                # This op can't be conv or conv_transpose
                continue

            bn_op = _match_pattern(op)
            if bn_op is not None:
                if self._try_to_transform(op, bn_op):
                    fusion_occurred = True
        return fusion_occurred


@register_pass(namespace="common")
class fuse_conv_bias(AbstractGraphPass):
    """
    Fold ``add``/``sub`` into ``bias`` of ``conv`` and ``conv_transpose``.
    That is, convert ``conv + add/sub`` to ``conv``, when ``add``/``sub`` is adding a constant.

    Two patterns are supported:

    .. code-block::

        Pattern 1:
        Given:
            %2 = conv(%1)
            ...
            %3 = add(%2, constant) # where constant has shape (1,C,1)/(C,1) for 1d conv, (1,C,1,1)/(C,1,1) for 2d conv etc
            ...

        Result:
            %3 = conv(%1)
            ...


        Pattern 2:
        Given:
            %2 = conv(%1)
            %3 = transpose(%2)
            ...
            %4 = add(%3, constant) # where constant has a broacasable shape
            ...

        Result:
            %2 = conv(%1)
            %4 = transpose(%2)
            ...
    """

    child_op_types = ["add", "sub"]

    def apply(self, prog):
        for f in prog.functions.values():
            block_changed = True
            while block_changed:
                block_changed = self._fuse_conv_bias_block(f)

    def _match_pattern(self, op):
        if op.op_type == "conv" or op.op_type == "conv_transpose":
            # abort fusion if op output is also a block output
            if op.outputs[0] in op.enclosing_block.outputs:
                return None
            # find add
            child_ops = op.outputs[0].child_ops
            if len(child_ops) == 1:
                add_op_candidate = list(child_ops)[0]
                if add_op_candidate.op_type in self.child_op_types:
                    return add_op_candidate
        return None

    @staticmethod
    def _try_to_transform_transpose_pattern(conv_op, block):

        ops_to_remove = []

        # conv layer
        if conv_op.op_type != "conv" and conv_op.op_type != "conv_transpose":
            return False
        is_deconv = conv_op.op_type == "conv_transpose"
        ops_to_remove.append(conv_op)

        # transpose layer
        if not _check_child_op_type(conv_op, "transpose"):
            return False
        transpose_op = list(conv_op.outputs[0].child_ops)[0]
        ops_to_remove.append(transpose_op)

        # add/sub layer
        if not _check_child_op_type(transpose_op, "add") and not _check_child_op_type(
            transpose_op, "sub"
        ):
            return False
        add_or_sub_op = list(transpose_op.outputs[0].child_ops)[0]

        ops_to_remove.append(add_or_sub_op)

        # get the bias
        if add_or_sub_op.x.val is None and add_or_sub_op.y.val is None:
            return False
        bias = add_or_sub_op.x.val if add_or_sub_op.x.val is not None else add_or_sub_op.y.val
        is_first_input = add_or_sub_op.y.val is not None
        is_sub = add_or_sub_op.op_type == "sub"

        # get the conv bias/weight
        conv_shape = conv_op.outputs[0].shape
        Cout = conv_shape[1]
        conv_weight = conv_op.weight.val
        conv_weight_type = conv_weight.dtype
        conv_bias = (
            np.zeros(Cout).astype(conv_weight_type) if conv_op.bias is None else conv_op.bias.val
        )

        # check if the bias is compatible for fusion
        is_bias_scalar = True
        if isinstance(bias, np.ndarray):
            if bias.shape == ():
                bias = bias.tolist()
            elif np.prod(bias.shape) == 1:
                bias = np.squeeze(bias).tolist()
            else:
                is_bias_scalar = False

        if not is_bias_scalar:
            if np.prod(bias.shape) != Cout:
                return False
            rank = transpose_op.outputs[0].rank
            cout_dim = transpose_op.perm.val.tolist().index(1) - rank
            if bias.shape[cout_dim] != Cout:
                return False
            bias = np.reshape(bias, (Cout))

        # compute the new bias
        if is_sub:
            if is_first_input:
                bias = -bias
            else:
                conv_bias = -conv_bias

        new_bias = conv_bias + bias

        # compute the new weight
        if is_sub and not is_first_input:
            new_weight = -conv_weight
        else:
            new_weight = conv_weight

        if not _check_no_output_connection(block, ops_to_remove):
            return False

        # create a new conv op with the new weight, bias value, copying rest of the attributes
        conv_kargs = {"weight": new_weight, "bias": new_bias, "before_op": conv_op}

        for k, v in conv_op.inputs.items():
            if k in ["weight", "bias"]:
                continue
            conv_kargs[k] = v

        if is_deconv:
            x = mb.conv_transpose(**conv_kargs)
        else:
            x = mb.conv(**conv_kargs)

        # create a new transpose op
        out_name = add_or_sub_op.outputs[0].name
        tranpose_kargs = {"x": x, "name": out_name, "before_op": transpose_op}
        for k, v in transpose_op.inputs.items():
            if k == "x":
                continue
            tranpose_kargs[k] = v
        x = mb.transpose(**tranpose_kargs)

        if add_or_sub_op.enclosing_block.try_replace_uses_of_var_after_op(
            anchor_op=add_or_sub_op,
            old_var=add_or_sub_op.outputs[0],
            new_var=x,
        ):
            add_or_sub_op.enclosing_block.remove_ops(ops_to_remove)
            return True
        return False

    @staticmethod
    def _try_to_transform(conv_op, add_op):

        if add_op.op_type == "sub":
            bias_var = add_op.y
        else:
            bias_var = add_op.x if add_op.x.val is not None else add_op.y
        bias_value = bias_var.val

        is_conv_op = conv_op.op_type == "conv"

        # check that the bias value is a constant array or a scalar constant
        if not isinstance(bias_value, (np.ndarray, np.generic)):
            return False

        is_bias_scalar = False
        if not isinstance(bias_value, np.ndarray):
            is_bias_scalar = True

        # find rank of the conv input
        rank = conv_op.x.rank
        if rank is None:
            return False
        if not (rank == 3 or rank == 4 or rank == 5):
            return False

        # check compatibility of bias value with the rank of the conv op
        # either bias value should be a scalar or:
        # rank=3 ==> (B,C,D), which means bias must be (1,C,1) or (C,1)
        # rank=4 ==> (B,C,D1,D2), which means bias must be (1,C,1,1) or (C,1,1)
        # rank=5 ==> (B,C,D1,D2,D3), which means bias must be (1,C,1,1,1) or (C,1,1,1)

        if is_bias_scalar:
            bias_value = np.array([bias_value])
        else:
            # check that there is at most one dimension in the shape that is not 1
            if len(np.squeeze(bias_value).shape) > 1:
                return False
            # check that addition is not happening on the batch dimension
            if len(bias_value.shape) == rank:
                if bias_value.shape[0] != 1:
                    return False
            # check that last rank-2 entries in the shape vector are all 1s
            if np.prod(bias_value.shape[-(rank - 2) :]) != 1:
                return False
            bias_value = np.squeeze(bias_value)

        if add_op.op_type == "sub":
            bias_value *= -1

        # everything looks good, now find the new updated bias
        old_bias = conv_op.inputs.get("bias", None)
        old_bias_value = None
        if old_bias is not None and old_bias.val is not None:
            old_bias_value = old_bias.val
        if old_bias is None:
            # need to create a fresh numpy array for bias
            if np.prod(bias_value.shape) == 1:
                # its a scalar bias
                # need to find the value of Cout to form a new bias
                if conv_op.weight.val is None:
                    return False
                # conv_transpose has weight format [K, C_out, spatial dims]
                # conv has weight format [C_out, K, spatial dims]
                Cout = conv_op.weight.val.shape[0 if is_conv_op else 1]
                new_bias_value = np.broadcast_to(bias_value, (Cout,))
            else:
                new_bias_value = bias_value
        else:
            # just need to update the existing bias array
            try:
                new_bias_value = old_bias_value + bias_value
            except:
                return False

        # create a new conv op with the new bias value, copying rest of the attributes
        out_name = add_op.outputs[0].name
        if new_bias_value.dtype != np.float32 and new_bias_value.dtype != np.float16:
            # cast the bias to match the weight type
            weight_np_type = types.nptype_from_builtin(
                conv_op.inputs["weight"].sym_type.get_primitive()
            )
            logger.warning(
                "conv_bias_fusion pass: casting bias "
                "from {} to {} to match the dtype of the weight of the conv layer".format(
                    new_bias_value.dtype, weight_np_type
                )
            )
            new_bias_value = new_bias_value.astype(weight_np_type)
        new_bias_var = mb.const(val=new_bias_value, before_op=conv_op)

        conv_kargs = {"bias": new_bias_var, "name": out_name, "before_op": conv_op}

        for k, v in conv_op.inputs.items():
            if k == "bias":
                continue
            conv_kargs[k] = v

        if is_conv_op:
            x = mb.conv(**conv_kargs)
        else:
            x = mb.conv_transpose(**conv_kargs)

        if add_op.enclosing_block.try_replace_uses_of_var_after_op(
            anchor_op=add_op,
            old_var=add_op.outputs[0],
            new_var=x,
        ):
            add_op.enclosing_block.remove_ops([conv_op, add_op])
            return True
        return False

    @block_context_manager
    def _fuse_conv_bias_block(self, block):
        fusion_occurred = False
        for op in list(block.operations):
            if op.enclosing_block is None:
                continue

            for b in op.blocks:
                block_changed = True
                while block_changed:
                    block_changed = self._fuse_conv_bias_block(b)
            if len(op.blocks) > 0:
                # This op can't be conv or conv_transpose
                continue

            # pattern 1 : conv + add/sub
            add_op = self._match_pattern(op)
            if add_op is not None:
                if self._try_to_transform(op, add_op):
                    fusion_occurred = True

            # pattern 2 : conv + transpose + add/sub
            elif self._try_to_transform_transpose_pattern(op, block):
                fusion_occurred = True

        return fusion_occurred


@register_pass(namespace="common")
class fuse_conv_scale(AbstractGraphPass):
    """
    Fold ``mul``/``div`` into ``conv``/``conv_transpose`` by updating the weight/bias of the convolution layers.

    The scale ``const`` can be a single number (scalar) or a vector with a broadcastable shape.
    For example, if the output of the ``conv``/``deconv`` layer is ``(B, Cout, H, W)``,
    ``const`` of shape ``(Cout, 1, 1)`` and ``(1, Cout, 1, 1)`` are allowed.

    .. code-block::

        Given:
            %2 = conv(%1)
            ...
            %3 = mul(%2, constant) # where constant is the scale constant
            ...

        Result:
            %3 = conv(%1)
            ...
    """

    def apply(self, prog):
        for f in prog.functions.values():
            block_changed = True
            while block_changed:
                block_changed = self._fuse_conv_scale_block(f)

    @staticmethod
    def _try_to_transform(conv_op, scale_op):
        # get the scale
        if scale_op.x.val is None and scale_op.y.val is None:
            return False
        scale_var = scale_op.x if scale_op.x.val is not None else scale_op.y
        scale = scale_var.val

        # for the scalar case, the scalar can be either
        # 1. a python int/float
        # 2. a 0d numpy array
        # 3. a 1d numpy array with shape (1,)

        is_scalar = True
        if isinstance(scale, np.ndarray):
            if scale.shape == ():
                scale = scale.tolist()
            elif scale.shape == (1) or scale.shape == (1,):
                scale = scale[0]
            else:
                is_scalar = False

        # get weight and bias and groups from conv layer
        if conv_op.weight.val is None:
            return False
        conv_weight = conv_op.weight.val
        conv_bias = conv_op.bias
        groups = conv_op.groups.val

        # get type of the conv layer
        is_deconv = conv_op.op_type == "conv_transpose"
        is_conv_1d = len(conv_weight.shape) == 3

        # D_in denotes the spatial dimensions for conv kernel weight
        # for conv_transpose, conv_weight has shape [Cin, Cout / groups, *D_in]
        # for conv, conv_weight has shape [Cout, Cin / groups, *D_in]
        if is_deconv:
            Cout = conv_weight.shape[1] * groups
            Cin = conv_weight.shape[0]
        else:
            Cout = conv_weight.shape[0]
            Cin = conv_weight.shape[1] * groups

        # for the vector scale case, check if the shape is broacastable
        if not is_scalar:
            if not np.prod(scale.shape) == Cout:
                return False
            if len(scale.shape) == len(conv_weight.shape):
                if not scale.shape[1] == Cout:
                    return False
            elif len(scale.shape) == len(conv_weight.shape) - 1:
                if not scale.shape[0] == Cout:
                    return False
            else:
                return False

        # transform the scale to 1./scale for the real_div case
        if scale_op.op_type == "real_div":
            scale = 1.0 / scale

        # get the type of the conv weight
        conv_weight_type = conv_weight.dtype

        # create bias for conv if not exist
        if conv_bias is None:
            conv_bias = np.zeros(Cout)
        else:
            conv_bias = conv_bias.val
        conv_bias = conv_bias.astype(conv_weight_type)

        # get the original shape of weight and bias
        origin_weight_shape = conv_weight.shape
        origin_bias_shape = conv_bias.shape

        # update the weight/bias for conv layer
        if is_scalar:
            new_conv_bias = np.array(conv_bias * scale).astype(conv_weight_type)
            new_conv_weight = np.array(conv_weight * scale).astype(conv_weight_type)

        else:
            scale = np.reshape(scale, (Cout))
            new_conv_bias = np.array(conv_bias * scale).astype(conv_weight_type)
            new_conv_weight = []
            if is_deconv:
                conv_weight = np.transpose(conv_weight, [1, 0, 2] if is_conv_1d else [1, 0, 2, 3])
                conv_weight = np.reshape(
                    conv_weight, [Cout, Cin // groups] + list(conv_weight.shape[2:])
                )

            for i in range(Cout):
                _conv_weight = conv_weight[i] * scale[i]
                new_conv_weight.append(_conv_weight)
            new_conv_weight = np.array(new_conv_weight).astype(conv_weight_type)

            if is_deconv:
                new_conv_weight = np.reshape(
                    new_conv_weight, [Cout // groups, Cin] + list(new_conv_weight.shape[2:])
                )
                new_conv_weight = np.transpose(
                    new_conv_weight, [1, 0, 2] if is_conv_1d else [1, 0, 2, 3]
                )

        # make sure the updated weight and bias have the same shape as the original ones
        assert (
            new_conv_weight.shape == origin_weight_shape
        ), "conv weight should have the same shape before and after the fuse_conv_scale pass."
        assert (
            new_conv_bias.shape == origin_bias_shape
        ), "conv bias should have the same shape before and after the fuse_conv_scale pass."

        # create a new conv op with the new weight, bias value, copying rest of the attributes
        out_name = scale_op.outputs[0].name
        conv_kargs = {
            "weight": new_conv_weight,
            "bias": new_conv_bias,
            "name": out_name,
            "before_op": conv_op,
        }

        for k, v in conv_op.inputs.items():
            if k in ["weight", "bias"]:
                continue
            conv_kargs[k] = v

        if is_deconv:
            x = mb.conv_transpose(**conv_kargs)
        else:
            x = mb.conv(**conv_kargs)

        if scale_op.enclosing_block.try_replace_uses_of_var_after_op(
            anchor_op=scale_op,
            old_var=scale_op.outputs[0],
            new_var=x,
        ):
            scale_op.enclosing_block.remove_ops([conv_op, scale_op])
            return True
        return False

    @block_context_manager
    def _fuse_conv_scale_block(self, block):
        def _match_pattern(op):
            if op.op_type == "conv" or op.op_type == "conv_transpose":
                # abort fusion if op output is also a block output
                if op.outputs[0] in op.enclosing_block.outputs:
                    return None
                # find batch_norm op
                child_ops = op.outputs[0].child_ops
                if len(child_ops) == 1:
                    scale_op_candidate = list(child_ops)[0]
                    if scale_op_candidate.op_type in ["mul", "real_div"]:
                        return scale_op_candidate
            return None

        fusion_occurred = False
        for op in list(block.operations):
            if op.enclosing_block is None:
                continue

            for b in op.blocks:
                block_changed = True
                while block_changed:
                    block_changed = self._fuse_conv_scale_block(b)
            if len(op.blocks) > 0:
                # This op can't be conv or conv_transpose
                continue

            scale_op = _match_pattern(op)

            if scale_op is not None:
                if self._try_to_transform(op, scale_op):
                    fusion_occurred = True

        return fusion_occurred


@register_pass(namespace="common")
class fuse_pad_conv(AbstractGraphPass):
    """
    When we observe ``pad -> transpose -> conv``, we move the ``pad`` to be next to ``conv``.
    This allows us to meld ``pad + conv`` if possible.

    .. code-block::

        Given:
            %1 = pad(%0, ...)
            %2 = transpose(%1, ...)
            %3 = conv(%2, ...)
            ...

        Result:
            %1.a = transpose(%0, ...)
            $2.a = pad(%1.a, ...)
            %3 = conv(%2.a)
            ...
    """

    def apply(self, prog):
        for f in prog.functions.values():
            block_changed = True
            while block_changed:
                block_changed = self._pad_conv_connect_block(f)

    @staticmethod
    def _match_pattern(op):
        ret = set([])
        child_ops = op.outputs[0].child_ops

        for child_op in child_ops:
            if child_op.op_type != "transpose":
                continue
            skip_ops = child_op.outputs[0].child_ops
            for skip_op in skip_ops:
                if "conv" not in skip_op.op_type:
                    continue
                ret.update([child_op])

        return ret if len(ret) != 0 else None

    @staticmethod
    def _try_to_transform(pad_op, transpose_ops, block):
        def _compute_new_pad_values(transpose_op):
            if pad_op.inputs["pad"].val is None:
                return None
            pad_amounts = np.reshape(pad_op.inputs["pad"].val, [-1, 2])
            transpose_axes = transpose_op.inputs["perm"].val
            rank_diff = len(transpose_axes) - pad_amounts.shape[0]
            pad_amounts_new = copy.deepcopy(pad_amounts)
            # append "rank_diff" rows of zeros to the top
            pad_amounts_new = np.concatenate(
                (np.zeros((2 * rank_diff)).reshape(-1, 2), pad_amounts_new)
            )
            pad_amounts_new = pad_amounts_new.astype(pad_amounts.dtype)
            pad_amounts = np.concatenate((np.zeros((2 * rank_diff)).reshape(-1, 2), pad_amounts))
            for i, axis in enumerate(transpose_axes):
                pad_amounts_new[i][0] = pad_amounts[axis][0]
                pad_amounts_new[i][1] = pad_amounts[axis][1]

            # get the top "rank_diff" rows
            top_rows = pad_amounts_new[:rank_diff, :]
            if not np.all(top_rows == 0):
                return False
            # cut "rank_diff" from the top
            pad_amounts_new = pad_amounts_new[rank_diff:, :]
            pad_amounts_new = pad_amounts_new.flatten()
            return pad_amounts_new

        if pad_op.outputs[0] in pad_op.enclosing_block.outputs:
            return False
        if len(set(pad_op.outputs[0].child_ops)) != len(transpose_ops):
            return False

        for transpose_op in transpose_ops:
            pad_amounts_new = _compute_new_pad_values(transpose_op)
            if pad_amounts_new is None:
                continue

            with pad_op.enclosing_block:
                new_transpose_var = mb.transpose(
                    x=pad_op.inputs["x"],
                    perm=transpose_op.inputs["perm"].val,
                    before_op=transpose_op,
                )
                new_pad_inputs = {"x": new_transpose_var, "pad": pad_amounts_new}
                for k, v in pad_op.inputs.items():
                    if k not in new_pad_inputs:
                        new_pad_inputs[k] = v
                new_pad_var = mb.pad(before_op=transpose_op, **new_pad_inputs)
            pad_op.enclosing_block.replace_uses_of_var_after_op(
                anchor_op=transpose_op, old_var=transpose_op.outputs[0], new_var=new_pad_var
            )

        pad_op.enclosing_block.remove_ops(list(transpose_ops) + [pad_op])

        return True

    @block_context_manager
    def _pad_conv_connect_block(self, block):
        fusion_occurred = False
        for op in list(block.operations):
            if op.enclosing_block is None:
                continue

            for b in op.blocks:
                block_changed = True
                while block_changed:
                    block_changed = self._pad_conv_connect_block(b)

            if op.op_type != "pad":
                continue

            transpose_ops = self._match_pattern(op)
            if transpose_ops is not None:
                if self._try_to_transform(op, transpose_ops, block):
                    fusion_occurred = True
        return fusion_occurred




@register_pass(namespace="common")
class fuse_dilated_conv(AbstractGraphPass):
    """
    When we observe ``space_to_batch -> conv (2D) -> batch_to_space``, we attempt to fuse these
    three ops into a single ``conv`` with dilations.

    .. code-block::

        Given:
            %1 = space_to_batch(%0, ...)
            %2 = conv(%1, ...)
            %3 = batch_to_space(%2, ...)
            ...

        Result:
            %3 = conv(%0, dilations=...)
            ...
    """

    @staticmethod
    def _uses_same_padding(
        input_h: int,
        input_w: int,
        W_h: int,
        W_w: int,
        dilation_factor: List[int],
        padding: List[int],
        crop: List[int],
    ) -> bool:
        base_paddings = [0] * 4

        dilated_W_h = dilation_factor[0] * (W_h - 1) + 1
        dilated_W_w = dilation_factor[1] * (W_w - 1) + 1

        base_paddings[0] = (dilated_W_h - 1) // 2
        base_paddings[1] = dilated_W_h - 1 - (dilated_W_h - 1) // 2
        base_paddings[2] = (dilated_W_w - 1) // 2
        base_paddings[3] = dilated_W_w - 1 - (dilated_W_w - 1) // 2

        pad_start_h = base_paddings[0]
        pad_start_w = base_paddings[2]
        orig_pad_end_h = base_paddings[1]
        orig_pad_end_w = base_paddings[3]
        full_input_h = input_h + pad_start_h + orig_pad_end_h
        full_input_w = input_w + pad_start_w + orig_pad_end_w
        pad_end_extra_h = (
            dilation_factor[0] - full_input_h % dilation_factor[0]
        ) % dilation_factor[0]
        pad_end_extra_w = (
            dilation_factor[1] - full_input_w % dilation_factor[1]
        ) % dilation_factor[1]
        pad_end_h = orig_pad_end_h + pad_end_extra_h
        pad_end_w = orig_pad_end_w + pad_end_extra_w

        return (
            padding[0] == pad_start_h
            and padding[1] == pad_end_h
            and padding[2] == pad_start_w
            and padding[3] == pad_end_w
            and crop[0] == 0
            and crop[1] == pad_end_extra_h
            and crop[2] == 0
            and crop[3] == pad_end_extra_w
        )

    def apply(self: AbstractGraphPass, prog: Program) -> None:
        for f in prog.functions.values():
            block_changed = True
            while block_changed:
                block_changed = self._fuse_dilated_conv_block(f)

    @staticmethod
    def _match_pattern(op: Operation) -> Optional[List[Operation]]:
        if op.op_type != "space_to_batch":
            return None

        if not _check_child_op_type(op, 'conv'):
            return None

        conv_op = op.outputs[0].child_ops[0]

        if len(conv_op.inputs['x'].shape[2:]) != 2:
            # restricted to Conv2d for now because in _try_to_transform function,
            # the logic for calculating whether padding is same or not, works only for 2d conv config.
            return None

        if not _check_child_op_type(conv_op, 'batch_to_space'):
            return None

        batch_to_space_op = conv_op.outputs[0].child_ops[0]

        return (op, conv_op, batch_to_space_op)

    @staticmethod
    def _try_to_transform(matched_ops: Tuple[Operation], block: Block) -> bool:

        if not _check_no_output_connection(block, matched_ops):
            return False

        space_to_batch_op, conv_op, batch_to_space_op = matched_ops

        stb_dilation_factor = space_to_batch_op.inputs['block_shape'].val
        bts_dilation_factor = batch_to_space_op.inputs['block_shape'].val

        if stb_dilation_factor is None or bts_dilation_factor is None:
            return False

        if list(stb_dilation_factor) != list(bts_dilation_factor):
            # If block_shape for space_to_batch and batch_to_space doesn't match,
            # we do not fuse.
            return False

        padding_val = space_to_batch_op.inputs['paddings'].val
        if padding_val is None:
            return False
        padding_val = padding_val.flatten()

        crop_val = batch_to_space_op.inputs['crops'].val
        if crop_val is None:
            return False
        crop_val = crop_val.flatten()

        has_same_padding = False
        if np.any(padding_val != 0):
            input_shape = space_to_batch_op.inputs['x'].shape
            W_shape = conv_op.inputs['weight'].shape
            W_h, W_w = W_shape[2], W_shape[3]
            HW = input_shape[2:]
            has_same_padding = fuse_dilated_conv._uses_same_padding(
                HW[0], HW[1], W_h, W_w, stb_dilation_factor, padding_val, crop_val
            )
            if not has_same_padding:
                return False

        conv_args = conv_op.inputs
        conv_args['x'] = space_to_batch_op.inputs['x']
        conv_args['dilations'] = list(stb_dilation_factor)
        if has_same_padding:
            conv_args['pad_type'] = 'same'

        new_var = mb.conv(**conv_args, before_op=conv_op)

        if conv_op.enclosing_block.try_replace_uses_of_var_after_op(
            anchor_op=conv_op, old_var=batch_to_space_op.outputs[0], new_var=new_var
        ):
            block.remove_ops(matched_ops)
            return True
        return False

    @block_context_manager
    def _fuse_dilated_conv_block(self: AbstractGraphPass, block: Block) -> bool:
        fusion_occurred = False
        for op in list(block.operations):
            if op.enclosing_block is None:
                continue

            for b in op.blocks:
                block_changed = True
                while block_changed:
                    block_changed = self._fuse_dilated_conv_block(b)

            matched_ops = self._match_pattern(op)
            if matched_ops is not None:
                if self._try_to_transform(matched_ops, block):
                    fusion_occurred = True
        return fusion_occurred
