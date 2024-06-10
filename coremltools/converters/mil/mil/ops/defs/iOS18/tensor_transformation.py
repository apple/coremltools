# Copyright (c) 2024, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import numpy as np

from coremltools.converters.mil.mil import Operation, types
from coremltools.converters.mil.mil.input_type import DefaultInputs, InputSpec, TensorInputType
from coremltools.converters.mil.mil.operation import Operation
from coremltools.converters.mil.mil.ops.defs._op_reqs import register_op
from coremltools.converters.mil.mil.ops.defs._utils import (
    get_param_val,
    solve_slice_by_index_shape,
    solve_slice_by_index_slice,
)
from coremltools.converters.mil.mil.ops.defs.iOS18 import _IOS18_TARGET
from coremltools.converters.mil.mil.types.symbolic import is_compatible_symbolic_vector


@register_op(opset_version=_IOS18_TARGET)
class slice_update(Operation):
    """
    Update a custom slice of a source tensor with another tensor of
    the same shape, as dictated by the slice.

    For example, if you have a tensor ``x``, this method produces the following::

        x[begin[0]: end[0]: stride[0], begin[1]: end[1]: stride[1], ...] = value

    The arguments defining the slice (``begin``, ``end``, ``stride``, ``masks``, and so on) should be
    treated the same way as iOS15 :py:class:`~.iOS15.tensor_transformation.slice_by_index`.


    Parameters
    ----------
    x: tensor<*?, T> (Required)
        * Input tensor.
    update: tensor<\*K, T> (Required)
        * Value tensor to be inserted.
        * The shape of the update tensor must match the slicing result of the input data.
        * rank-0 update is not supported.
    begin: tensor<[rank<x>], U> (Required)
        * Starting index for the dimension of slicing.
    end: tensor<[rank(x)], U> (Required)
        * Ending index for the dimension of slicing.
    stride: tensor<[rank(x)], U> (Optional)
        * Default as all ``1``.
        * Stride for the dimension of slicing.
    begin_mask: tensor<[rank(x)], bool> (Optional)
        * Default to all ``False``.
        * If ``begin_mask[i]==True``, neglect ``begin[i]``, and set ``begin[i]`` to ``0``.
    end_mask: tensor<[rank(x)], bool> (Optional)
        * Default to all ``False``.
        * If ``end_mask[i]==True``, neglect ``end[i]``, and set ``end[i]`` to ``x.shape[i]``.
    squeeze_mask: tensor<[rank(x)], bool> (Optional)
        * Default to all ``False``.
        * If ``squeeze_mask[i]==True``, neglect ``end[i]``, and do the pure index at ``begin[i]``.

    Returns
    -------
    tensor<\*?, T>
        - Scalar or tensor.

    Attributes
    ----------
    T: fp16, fp32, int8, int16, int32, uint8, uint16, bool
    U: int8, int16, int32
    """

    input_spec = InputSpec(
        x=TensorInputType(type_domain="T"),
        update=TensorInputType(type_domain="T"),
        begin=TensorInputType(type_domain="U"),
        end=TensorInputType(type_domain="U"),
        stride=TensorInputType(const=True, optional=True, type_domain="U"),
        begin_mask=TensorInputType(const=True, optional=True, type_domain=types.bool),
        end_mask=TensorInputType(const=True, optional=True, type_domain=types.bool),
        squeeze_mask=TensorInputType(const=True, optional=True, type_domain=types.bool),
    )

    type_domains = {
        "T": (
            types.fp16,
            types.fp32,
            types.int8,
            types.int16,
            types.int32,
            types.uint8,
            types.uint16,
            types.bool,
        ),
        "U": (types.int8, types.int16, types.int32),
    }

    def default_inputs(self):
        return DefaultInputs(
            stride=None,
            begin_mask=None,
            end_mask=None,
            squeeze_mask=None,
        )

    def type_inference(self):
        # solve shape
        ret_shape = solve_slice_by_index_shape(
            self.x.shape,
            self.begin.val,
            self.end.val,
            get_param_val(self.stride),
            get_param_val(self.begin_mask),
            get_param_val(self.end_mask),
            get_param_val(self.squeeze_mask),
        )

        if not is_compatible_symbolic_vector(ret_shape, self.update.shape):
            raise ValueError(
                "The update tensor should have shape {}. Got {}".format(
                    ret_shape, self.update.shape
                )
            )

        if self.update.rank == 0:
            # rdar://128221986 ([Feature][Slice_update] The backends is not supporting scalar update for the slice_update op)
            raise ValueError(f"rank-0 'update' is not supported in 'slice_update' op {self.name}.")

        return self.x.sym_type

    def value_inference(self):
        if (
            self.x.sym_val is None
            or self.update.sym_val is None
            or self.begin.val is None
            or self.end.val is None
        ):
            return None

        # solve the data slices
        slices = solve_slice_by_index_slice(
            self.x.shape,
            self.begin.val,
            self.end.val,
            get_param_val(self.stride),
            get_param_val(self.begin_mask),
            get_param_val(self.end_mask),
            get_param_val(self.squeeze_mask),
        )

        # copy the data and do the inplace update
        copy_x_val = np.copy(self.x.sym_val)
        copy_x_val[slices] = np.reshape(self.update.sym_val, copy_x_val[slices].shape)
        return copy_x_val
