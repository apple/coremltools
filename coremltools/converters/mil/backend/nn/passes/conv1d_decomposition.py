#  Copyright (c) 2023, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from coremltools.converters.mil.mil import Block
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import Operation
from coremltools.converters.mil.mil.passes.graph_pass import AbstractGraphPass
from coremltools.converters.mil.mil.passes.helper import block_context_manager
from coremltools.converters.mil.mil.passes.pass_registry import register_pass


@register_pass(namespace="nn_backend")
class decompose_conv1d(AbstractGraphPass):
    """
    NeuralNetwork does not support conv1d natively,
    instead it decomposes conv1d into expand_dims -> conv2d -> squeeze

    Let us decompose conv1d for NN,
    so we may have a chance to optimize expand_dims -> conv2d -> squeeze

    Given:
        %2 = conv(%1), %1.rank = 3
        ...

    Result:
        %3 = expand_dims(%1, axes=-2)
        %4 = conv(%3)
        %2 = squeeze(%4, axes=-2)
        ...

    """

    def apply(self, prog):
        for f in prog.functions.values():
            self._decompose_conv1d_block(f)

    @block_context_manager
    def _decompose_conv1d_block(self, block: Block):
        def help_decompose_conv1d_block(block: Block) -> bool:
            fusion_occurred = False
            for op in list(block.operations):
                if op.enclosing_block is None:
                    continue

                for b in op.blocks:
                    block_changed = True
                    while block_changed:
                        block_changed = help_decompose_conv1d_block(b)

                # must be conv1d
                if op.op_type != "conv" or op.x.rank != 3:
                    continue

                with mb.set_before_op(op):
                    if self._try_apply_transform(op, block):
                        fusion_occurred = True

            return fusion_occurred

        block_changed = True
        while block_changed:
            block_changed = help_decompose_conv1d_block(block)

    @staticmethod
    def _try_apply_transform(conv_op: Operation, block: Block) -> bool:
        # create `expand_dims`
        expand_out = mb.expand_dims(x=conv_op.x, axes=(-2,))

        # prepare `conv2d`
        conv_kwargs = {"x": expand_out}

        # inherit `pad_type`, `groups`, `bias` from `conv1d`
        conv_kwargs["pad_type"] = conv_op.inputs["pad_type"].val
        conv_kwargs["groups"] = conv_op.inputs["groups"].val
        bias = conv_op.inputs.get("bias", None)
        if bias is not None:
            conv_kwargs["bias"] = bias

        # expand `weight`, `strides`, `pad`, `dilations` from `conv1d`
        conv_kwargs["weight"] = mb.expand_dims(x=conv_op.inputs["weight"], axes=(-2,))
        conv_kwargs["strides"] = (1, conv_op.inputs["strides"].val[-1])
        conv_kwargs["pad"] = (0, 0, conv_op.inputs["pad"].val[-2], conv_op.inputs["pad"].val[-1])
        conv_kwargs["dilations"] = (1, conv_op.inputs["dilations"].val[-1])

        # compose `conv2d`
        conv_out = mb.conv(**conv_kwargs)

        # create `squeeze`
        squeeze_out = mb.squeeze(x=conv_out, axes=(-2,), name=conv_op.outputs[0].name)

        # try replacing `conv1d` output
        # with the new `expand_dims` -> `conv2d` -> `squeeze` output
        if conv_op.enclosing_block.try_replace_uses_of_var_after_op(
            anchor_op=conv_op, old_var=conv_op.outputs[0], new_var=squeeze_out
        ):
            # remove `conv1d`
            block.remove_ops([conv_op])
            return True
        return False
