#  Copyright (c) 2023, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from typing import Tuple

import numpy as np

import coremltools.converters.mil.mil.types as types
from coremltools.converters.mil._deployment_compatibility import AvailableTarget
from coremltools.converters.mil.mil import Block
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import Operation, Var
from coremltools.converters.mil.mil.passes.graph_pass import AbstractGraphPass
from coremltools.converters.mil.mil.passes.helper import (
    _check_child_op_type,
    _check_no_output_connection,
    block_context_manager,
)
from coremltools.converters.mil.mil.passes.pass_registry import register_pass


@register_pass(namespace="common")
class int_op_canonicalization(AbstractGraphPass):
    """
    For general quantized operators, in Core ML, we represent them as
    ``dequantize -> the floating-point version of this operator -> quantize``,
    because mathematically it is the floating-point tensor rather than
    its quantized integer representation that gets operated upon.

    For some quantized operators that do not involve floating-point arithmetic,
    however, it is unnecessary to prepend ``dequantize`` and append ``quantize``.
    Examples are:

    * reshape
    """

    INT_OP_TYPES_AND_OPSET_VERSIONS = {"reshape": {AvailableTarget.iOS17}}

    def apply(self, prog):
        for f in prog.functions.values():
            self._canonicalize_int_ops_block(f)

    @block_context_manager
    def _canonicalize_int_ops_block(self, block: Block):
        def apply_block(block: Block) -> bool:
            for op in list(block.operations):
                for b in op.blocks:
                    self._canonicalize_int_ops_block(b)

                matched_ops = self.match_pattern(op)
                if matched_ops is not None:
                    dequantize, quantize = matched_ops
                    # has to break as the downstream iterator is affected
                    if self.try_to_transform(dequantize, op, quantize):
                        return True

            return False

        need_transformation = True
        while need_transformation:
            need_transformation = apply_block(block)

    def match_pattern(self, op: Operation) -> Tuple[Operation, Operation]:
        if (
            op.op_type not in self.INT_OP_TYPES_AND_OPSET_VERSIONS
            or op.opset_version not in self.INT_OP_TYPES_AND_OPSET_VERSIONS[op.op_type]
        ):
            return None

        # make sure the input is quantized
        dequantize = op.x.op
        if dequantize is None or dequantize.op_type != "dequantize":
            return None

        # make sure the output is quantized
        if not _check_child_op_type(op, "quantize"):
            return None
        quantize = op.outputs[0].child_ops[0]

        # we do not have to check block output, because:
        # * for dequantize, it is ok to connect to block output, since our
        #   transformation method `try_to_transform` is able to deal with that
        # * for op, checking child op has made sure it has only 1 child
        #   and connects to quantize, i.e. it cannot connect to block output

        return dequantize, quantize

    def try_to_transform(self, dequantize: Operation, op: Operation, quantize: Operation) -> bool:
        block: Block = op.enclosing_block

        if not block.try_replace_uses_of_var_after_op(
            anchor_op=quantize,
            old_var=quantize.outputs[0],
            new_var=self.build_int_op(dequantize, op, quantize),
        ):
            return False

        # remove op and quantize here, but not dequantize, since:
        # * all uses of op and quantize has been replaced with the canonicalized one
        # * dequantize may feed to multiple ops, which are not replaced
        #   (if not, then pass dead_code_elimination will eliminate it)
        block.remove_ops([op, quantize])

        return True

    @staticmethod
    def build_int_op(dequantize: Operation, op: Operation, quantize: Operation) -> Var:
        if op.op_type == "reshape":
            return mb.reshape(
                x=dequantize.input,
                shape=op.shape,
                name=quantize.outputs[0].name,
                before_op=op,
            )

        raise NotImplementedError(f"no build method implemented for int op {op.op_type}")


# TODO (rdar://107718371): remove this pass after implementing QuantizedVar
@register_pass(namespace="common")
class nullify_redundant_quantization_zero_point(AbstractGraphPass):
    """
    In Core ML quantization, the performance is better when ``zero point = 0``,
    so we try to make ``zero point = 0`` if possible:

    * ``zero point = -128``
        * this must be an int8 quantization
        * equivalent to uint8 quantization with 0 zero point
    * ``zero point = 128``
        * this must be an uint8 quantization
        * equivalent to int8 quantization with 0 zero point

    Since ``zero point = 0`` is equivalent to ``zero point = None`` in Core ML semantics,
    we further canonicalize to ``zero point = None`` to:

    * make further graph passes easier
    * avoid serializing trivial 0

    The ``zero point = 0`` case can be canonicalized trivially

    .. code-block::

        Input op:

            quantize/dequantize(zero_point=0)

        Output op:

            quantize/dequantize(zero_point=None)

    To guarantee the conservation of output regardless the zero-point shift
    in ``zero point = ±128`` cases, we would only transform:

    * const dequantize, where we fuse the zero-point shift into the const

    .. code-block::

        Input op:

            dequantize(input=const, zero_point=±128)

        Output op:

            dequantize(input=const∓128, zero_point=None)

    * ``quantize -> dequantize``, where we nullify both simultaneously

    .. code-block::

        Input graph:

            input -> quantize(zero_point=±128) -> dequantize(zero_point=±128) -> output

        Output graph:

            input -> quantize(zero_point=None) -> dequantize(zero_point=None) -> output
    """

    def apply(self, prog):
        for f in prog.functions.values():
            self._nullify_redundant_quantization_zero_point_block(f)

    @block_context_manager
    def _nullify_redundant_quantization_zero_point_block(self, block: Block):
        def apply_block(block: Block) -> bool:
            for op in list(block.operations):
                for b in op.blocks:
                    self._nullify_redundant_quantization_zero_point_block(b)

                # no need to break, since only the current op gets changed
                self.try_transform_zp0(op)
                self.try_transform_zp128_const_dequantize(op)

                # has to break as the downstream iterator is affected
                if self.try_transform_zp128_quantize_dequantize(op):
                    return True

            return False

        need_transformation = True
        while need_transformation:
            need_transformation = apply_block(block)

    @staticmethod
    def try_transform_zp0(op: Operation) -> bool:
        if op.op_type not in ("quantize", "dequantize"):
            return False

        zero_point = op.zero_point
        # if already no zero point, no need for further nullification
        if zero_point is None:
            return False
        zero_point = zero_point.val

        if not np.all(zero_point == 0):
            return False

        new_var: Var
        if op.op_type == "quantize":
            new_var = mb.quantize(
                input=op.input,
                scale=op.scale,
                axis=op.axis,
                output_dtype=op.output_dtype,
                before_op=op,
            )
        else:
            new_var = mb.dequantize(
                input=op.input,
                scale=op.scale,
                axis=op.axis,
                before_op=op,
            )

        block: Block = op.enclosing_block
        if not block.try_replace_uses_of_var_after_op(
            anchor_op=op, old_var=op.outputs[0], new_var=new_var
        ):
            return False
        block.remove_ops([op])

        return True

    @staticmethod
    def try_transform_zp128_const_dequantize(op: Operation) -> bool:
        if op.op_type != "dequantize":
            return False

        zero_point = op.zero_point
        # if already no zero point, no need for further nullification
        if zero_point is None:
            return False
        zero_point = zero_point.val

        is_negative_128 = np.all(zero_point == -128)
        is_positive_128 = np.all(zero_point == 128)
        if not (is_negative_128 or is_positive_128):
            return False

        input = op.input.val
        if input is None:
            return False
        if is_negative_128:
            input = np.uint8(np.int16(input) + 128)
        else:
            input = np.int8(np.int16(input) - 128)

        new_var = mb.dequantize(
            input=input,
            scale=op.scale,
            axis=op.axis,
            before_op=op,
        )

        block: Block = op.enclosing_block
        if not block.try_replace_uses_of_var_after_op(
            anchor_op=op, old_var=op.outputs[0], new_var=new_var
        ):
            return False
        block.remove_ops([op])

        return True

    @staticmethod
    def try_transform_zp128_quantize_dequantize(op: Operation) -> bool:
        if op.op_type != "quantize":
            return False

        zero_point = op.zero_point
        # if already no zero point, no need for further nullification
        if zero_point is None:
            return False
        zero_point = zero_point.val

        is_negative_128 = np.all(zero_point == -128)
        is_positive_128 = np.all(zero_point == 128)
        if not (is_negative_128 or is_positive_128):
            return False

        if not _check_child_op_type(op, "dequantize"):
            return False
        dequantize_op = op.outputs[0].child_ops[0]

        dequantize_zero_point = dequantize_op.zero_point
        if dequantize_zero_point is None:
            return False
        dequantize_zero_point = dequantize_zero_point.val

        if not np.all(dequantize_zero_point == (-128 if is_negative_128 else 128)):
            return False

        new_quantize = mb.quantize(
            input=op.input,
            scale=op.scale,
            axis=op.axis,
            output_dtype="uint8" if is_negative_128 else "int8",
            before_op=dequantize_op,
        )
        new_dequantize = mb.dequantize(
            input=new_quantize,
            scale=dequantize_op.scale,
            axis=dequantize_op.axis,
            before_op=dequantize_op,
        )

        block: Block = op.enclosing_block
        if not block.try_replace_uses_of_var_after_op(
            anchor_op=dequantize_op,
            old_var=dequantize_op.outputs[0],
            new_var=new_dequantize,
        ):
            return False
        block.remove_ops([op, dequantize_op])
        return True


@register_pass(namespace="common")
class dequantize_quantize_pair_elimination(AbstractGraphPass):
    """
    When a ``dequantize`` is followed by an identical ``quantize`` (same scale,
    zero point, axis), they cancel out and can be eliminated

    .. code-block::

        Input graph:
            input -> dequantize -> quantize -> output

        Output graph:
            input -> output

    PS: On the other hand, the reversed pattern, i.e., ``quantize -> dequantize``,
    is not redundant, since that is the pattern which naturally occurs when a
    quantized op is converted.
    In current activation quantization conversion, a quantized op becomes

    .. code-block::

        dequantize -> regular op -> quantize

    so if we have a sequence of quantized ops, we will get

    .. code-block::

        dequantize -> regular op1 -> quantize -> dequantize -> regular op2 -> quantize

    The ``quantize -> dequantize`` pair in the middle is not redundant, even if
    they have identical scales and zero points and axes, since removing them will lead to
    loss of information about the quantization parameters of the output var of op1
    """

    def apply(self, prog):
        for f in prog.functions.values():
            self._dequantize_quantize_pair_elimination_block(f)

    @block_context_manager
    def _dequantize_quantize_pair_elimination_block(self, block):
        def apply_block(block: Block) -> bool:
            for op in list(block.operations):
                for b in op.blocks:
                    self._dequantize_quantize_pair_elimination_block(b)

                # has to break as the downstream iterator is affected
                if self.try_dequantize_quantize_pair_elimination(op):
                    return True

            return False

        need_transformation = True
        while need_transformation:
            need_transformation = apply_block(block)

    @staticmethod
    def try_dequantize_quantize_pair_elimination(op: Operation) -> bool:
        if op.op_type != "dequantize":
            return False

        if op.outputs[0] in op.enclosing_block.outputs:
            return False

        if not _check_child_op_type(op, "quantize"):
            return False
        quantize_op = op.outputs[0].child_ops[0]

        if np.any(op.scale.val != quantize_op.scale.val):
            return False

        is_dequantize_zp_present = op.zero_point is not None
        is_quantize_zp_present = quantize_op.zero_point is not None
        if is_dequantize_zp_present != is_quantize_zp_present:
            return False
        if is_dequantize_zp_present and is_quantize_zp_present:
            if np.any(op.zero_point.val != quantize_op.zero_point.val):
                return False

        is_dequantize_axis_present = op.axis is not None
        is_quantize_axis_present = quantize_op.axis is not None
        if is_dequantize_axis_present != is_quantize_axis_present:
            return False
        if is_dequantize_axis_present and is_quantize_axis_present:
            if op.axis.val != quantize_op.axis.val:
                return False

        block: Block = op.enclosing_block
        if not block.try_replace_uses_of_var_after_op(
            anchor_op=quantize_op,
            old_var=quantize_op.outputs[0],
            new_var=op.input,
        ):
            return False
        block.remove_ops([op, quantize_op])
        return True


@register_pass(namespace="common")
class distributive_quantized_binary_op_scale_normalization(AbstractGraphPass):
    """
    In the backend, for better performance, quantized op can have 1 input scale
    fused within the quantized op kernel. For binary ops, there are 2 inputs,
    but only 1 can get fused. For example, for quantized ``add``

    .. code-block::

        MIL graph (consists of MIL ops):

            dequantize(x, s_x, zp_x) -|
            x_fp = (x - zp_x) * s_x   |
                                      |->  add(x_fp, y_fp)   -> quantize(z_fp, s_z, zp_z)
            dequantize(y, s_y, zp_y) -|   z_fp = x_fp + y_fp      z = z_fp / s_z + zp_z
            y_fp = (y - zp_y) * s_y

        Backend graph (consists of backend instructions, usually including + - * / and fused *+):

            x_shift = x - zp_x -------------------------|
                                                        |-> z_fp = s_x * x_shift + y_fp -> z = z_fp / s_z + zp_z
            y_shift = y - zp_y -> y_fp = s_y * y_shift -|

    Where ``x`` and ``y`` are the inputs, ``z`` is the output,
    ``s`` and ``zp`` are the corresponding scale and zero point.

    The reason why fusing one scale leads to better performance is,
    instead of 2 instructions ``x_fp = s_x * x_shift`` and ``z_fp = x_fp + y_fp``,
    a single ``z_fp = x_shift * s_x + y_fp`` instruction achieves the same result.

    In this pass, we normalize ``s_y`` to 1, so the ``y_fp = s_y * y_shift``
    instruction can get skipped as well, leading to even better performance.
    This pass only applies to distributive binary ops such as ``add`` and ``sub``

    Appendix: Mathematical and Computer-Scientific Details

    Mathematically, for a binary operator ``.op.``

    .. code-block::

        z_fp = (x - zp_x) * s_x .op. (y - zp_y) * s_y
             = s_y * [(x - zp_x) * s_x/s_y .op. (y - zp_y) * 1]

    The corresponding pseudo code is

    .. code-block::

        # before
        z_fp = (x - zp_x) * s_x .op. (y - zp_y) * s_y
        z = z_fp / s - zp_z

        # after
        z_fp_modified = (x - zp_x) * s_x/s_y .op. (y - zp_y) * 1.0
        z = z_fp_modified / (s_z/s_y) - zp_z

    Concretely, as a MIL graph pass

    .. code-block::

        Input graph:
            dequantize(scale=s_x) -|
                                   |-> op -> quantize(scale=s_z)
            dequantize(scale=s_y) -|

        Output graph:
            dequantize(scale=s_x/s_y) -|
                                       |-> op -> quantize(scale=s_z/s_y)
            dequantize(scale=1.0)     -|

    PS: we only support scalar ``s_y`` for now. If ``s_y`` is not scalar but
    ``s_x`` is, we would swap ``x`` and ``y``. Support for both-vector case is
    to be explored, due to the broadcasting complication.
    """

    DISTRIBUTIVE_BINARY_OPS = {"add", "sub"}

    def apply(self, prog):
        @block_context_manager
        def apply_block(block: Block):
            for op in list(block.operations):
                for b in op.blocks:
                    apply_block(b)

                matched_ops = self.match_pattern(op)
                if matched_ops is not None:
                    dequantize_x, dequantize_y, quantize_z = matched_ops
                    self.try_to_transform(op, dequantize_x, dequantize_y, quantize_z)

        for f in prog.functions.values():
            apply_block(f)

    def match_pattern(self, op: Operation) -> Tuple[Operation, Operation, Operation]:
        """
        try to match distributive quantized binary op:
                ...
                 ^
                 |
            dequantize(x) -|
                           |-> op(x, y) (-> relu) -> quantize(z)
            dequantize(y) -|
                 |
                 v
                ...

        return dequantize_x, dequantize_y, quantize_z for further transformation

        return None if no match
        """
        # make sure the op is distributive
        if op.op_type not in self.DISTRIBUTIVE_BINARY_OPS:
            return None

        # quantized op may be fused with relu
        # relu would not affect distributivity
        tail_op = op
        if _check_child_op_type(op, "relu"):
            tail_op = op.outputs[0].child_ops[0]

        # make sure the inputs are quantized
        dequantize_x = op.x.op
        dequantize_y = op.y.op
        if (
            dequantize_x is None
            or dequantize_y is None
            or dequantize_x.op_type != "dequantize"
            or dequantize_y.op_type != "dequantize"
        ):
            return None

        # make sure the output is quantized
        if not _check_child_op_type(tail_op, "quantize"):
            return None
        quantize_z = tail_op.outputs[0].child_ops[0]

        # make sure the intermediate results are not block outputs
        # since we only guarantee conservation of z
        if not _check_no_output_connection(
            op.enclosing_block, [dequantize_x, dequantize_y, op, tail_op, quantize_z]
        ):
            return None

        return dequantize_x, dequantize_y, quantize_z

    def try_to_transform(
        self, op: Operation, dequantize_x: Operation, dequantize_y: Operation, quantize_z: Operation
    ) -> bool:
        """
        given dequantize_x, dequantize_y, quantize_z, tranform by
            z_fp = (x - zp_x) * s_x/s_y .op. (y - zp_y) * 1.0
            z = z_fp / (s_z/s_y) - zp_z

        See the class doc for details
        """
        block = quantize_z.enclosing_block

        new_s_x, new_s_z = self.try_to_divide(dequantize_x, dequantize_y, quantize_z)
        # if s_y cannot be used to divide, then swap x and y and try again
        if new_s_x is None and new_s_z is None:
            dequantize_x, dequantize_y = dequantize_y, dequantize_x
            new_s_x, new_s_z = self.try_to_divide(dequantize_x, dequantize_y, quantize_z)
            # after swap, if still cannot divide, then give up
            if new_s_x is None and new_s_z is None:
                return False

        # insert normalized new_dequantize_x and new_dequantize_y before op
        new_dequantize_x = mb.dequantize(
            input=dequantize_x.input,
            scale=new_s_x,
            zero_point=dequantize_x.zero_point,
            axis=dequantize_x.axis,
            before_op=op,
        )
        new_dequantize_y = mb.dequantize(
            input=dequantize_y.input,
            scale=1.0 if dequantize_y.axis is None else np.full(dequantize_y.scale.val.shape, 1.0),
            zero_point=dequantize_y.zero_point,
            axis=dequantize_y.axis,
            before_op=op,
        )
        # insert normalized new_quantize_z before quantize_z
        new_quantize_z = mb.quantize(
            input=quantize_z.input,
            scale=new_s_z,
            zero_point=quantize_z.zero_point,
            axis=quantize_z.axis,
            output_dtype=quantize_z.output_dtype,
            before_op=quantize_z,
        )
        if not (
            # replace dequantize_x and dequantize_y with the normalized ones
            # in the range of (new_dequantize_x, op] and (new_dequantize_y, op]
            # in case dequantize_x and dequantize_y also feed to other ops
            # which should not get altered by this transformation
            block.try_replace_uses_of_var_after_op(
                anchor_op=new_dequantize_x.op,
                end_op=op,
                old_var=dequantize_x.outputs[0],
                new_var=new_dequantize_x,
            )
            and block.try_replace_uses_of_var_after_op(
                anchor_op=new_dequantize_y.op,
                end_op=op,
                old_var=dequantize_y.outputs[0],
                new_var=new_dequantize_y,
            )
            # replace quantize_z with the normalized one
            and block.try_replace_uses_of_var_after_op(
                anchor_op=quantize_z, old_var=quantize_z.outputs[0], new_var=new_quantize_z
            )
        ):
            return False

        # remove quantize_z here, but not dequantize_x and dequantize_y, since:
        # * all uses of quantize_z has been replaced with the normalized one
        # * dequantize_x and dequantize_y may feed to multiple ops, which are not replaced
        #   (if not, then pass dead_code_elimination will eliminate them)
        block.remove_ops([quantize_z])

        return True

    def try_to_divide(
        self,
        dequantize_x: Operation,
        dequantize_y: Operation,
        quantize_z: Operation,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        compute s_x/s_y and s_z/s_y, return the results if succeeds, else None

        The broadcast rule is very complicated:
        1. Broadcast s_x to x, s_y to y, s_z to z, according to axes
        2. Broadcast s_x and s_y
        3. Perform s_x/s_y and s_z/s_y
        4. De-broadcast s_x/s_y and s_z/s_y down to vectors according to axes,
           raise exception if impossible to de-broadcast

        As a result, for now we only handle the scalar s_y case
        """

        # TODO (rdar://109170887): explore vector s_y
        if dequantize_y.axis is not None:
            return None, None

        s_x_fp32 = np.float32(dequantize_x.scale.val)
        s_y_fp32 = np.float32(dequantize_y.scale.val)
        s_z_fp32 = np.float32(quantize_z.scale.val)

        s_x_d_s_y = s_x_fp32 / s_y_fp32
        s_z_d_s_y = s_z_fp32 / s_y_fp32

        if (
            self.overflow_fp16(s_x_d_s_y)
            or self.underflow_fp16(s_x_d_s_y)
            or self.overflow_fp16(s_z_d_s_y)
            or self.underflow_fp16(s_z_d_s_y)
        ):
            return None, None

        return s_x_d_s_y, s_z_d_s_y

    @staticmethod
    def overflow_fp16(x: np.ndarray) -> bool:
        return np.max(np.abs(x)) > 65504

    @staticmethod
    def underflow_fp16(x: np.ndarray) -> bool:
        return np.min(np.abs(x)) < np.nextafter(0.0, 1.0, dtype=np.float16)


@register_pass(namespace="common")
class dequantize_to_constexpr(AbstractGraphPass):
    """
    ``dequantize`` op with constant input is equivalent to ``constexpr_affine_dequantize``.
    This is one of the canoncalization pass that transforms all such
    ``dequantize`` ops to respective ``constexpr_affine_dequantize`` ops.

    .. code-block::

        Input graph:

            dequantize(input=const) -> downstream op

        Output graph:

            constexpr_affine_dequantize -> downstream op

    This pass is being performed because constant tensors being propagated
    through ``dequantize`` op would be serialized in bloated/decompressed fashion,
    whereas with ``constexpr_affine_dequantize``,
    constant weights/tensors remain compressed at serialization.
    """

    def apply(self, prog):
        @block_context_manager
        def apply_block(block):
            for op in list(block.operations):
                for b in op.blocks:
                    apply_block(b)

                if self.is_valid_op(op):
                    self.transform_op(op)

        for f in prog.functions.values():
            apply_block(f)

    def is_valid_op(self, op):
        return op.op_type == "dequantize" and op.outputs[0].val is not None

    def transform_op(self, op):
        quantized_data = op.input.val

        scale = op.scale.val

        zero_point = None
        if op.zero_point is not None:
            zero_point = op.zero_point.val
        else:
            zero_point = np.int8(0) if op.input.dtype == types.int8 else np.uint8(0)

        # In dequantize semantics, axis may be None:
        #     when scale is a scalar, axis is None
        #
        # In constexpr_affine_dequantize semantics, None axis is not allowed;
        # since axis is not referred to when scale is a scalar, we pass a dummy
        axis = 0
        if op.axis is not None:
            axis = op.axis.val

        new_var = mb.constexpr_affine_dequantize(
            quantized_data=quantized_data,
            zero_point=zero_point,
            scale=scale,
            axis=axis,
            before_op=op,
            name=op.name + "_affine_dequantized",
        )

        block = op.enclosing_block
        block.replace_uses_of_var_after_op(anchor_op=op, old_var=op.outputs[0], new_var=new_var)
        block.remove_ops([op])
