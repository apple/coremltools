#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause
import numpy as np

from coremltools.converters.mil.mil import (
    BoolInputType,
    DefaultInputs,
    InputSpec,
    Operation,
    precondition,
    StringInputType,
    TensorInputType,
    TupleInputType,
    types,
)
from coremltools.converters.mil.mil.operation import VALUE
from coremltools.converters.mil.mil.types.symbolic import is_symbolic
from ._op_reqs import register_op
from ._utils import broadcast_shapes, parse_einsum_equation

@register_op(doc_str="")
class linear(Operation):
    """
    Perform  ``x * weight.T + bias`` where ``weight`` and ``bias`` are constant at
    compile time.

    Parameters
    ----------
    x: tensor<[\*D,D_in], T> (Required)
        * ``1 <= rank <= 3``.
        * ``0 <= rank(*D) <= 2``.
    weight: const tensor<[D_out,D_in], T> (Required)
    bias: const tensor<[D_out],T> (Optional)
        * Default to ``0``.

    Returns
    -------
    tensor<[\*D,D_out], T>
        * Same rank as the input ``x``.

    Attributes
    ----------
    T: fp32
    """
    input_spec = InputSpec(
        x=TensorInputType(),
        weight=TensorInputType(const=True),
        bias=TensorInputType(const=True, optional=True),
    )

    def default_inputs(self):
        Dout = self.weight.shape[0]
        return DefaultInputs(
            bias=[0.]*Dout,
            )

    def __init__(self, **kwargs):
        super(linear, self).__init__(**kwargs)

    def type_inference(self):
        x_type = self.x.dtype
        x_shape = self.x.shape
        weight_shape = self.weight.shape
        assert len(weight_shape) == 2

        shape = list(x_shape)
        shape[-1] = weight_shape[0]
        return types.tensor(x_type, tuple(shape))

    @precondition(allow=VALUE)
    def value_inference(self):
        res = np.matmul(self.x.val, np.transpose(self.weight.val))
        if self.bias is not None:
            res += self.bias.val
        return res


@register_op(doc_str="")
class matmul(Operation):
    """
    Perform N-D batch matrix multiplication with NumPy-style broadcasting
    based on the following rules:

    Rule 1. If both ``x, y`` are 1-D, return the scalar from the dot product.

    Rule 2. If both ``x, y`` are 2-D or higher, perform a broadcast on the batch dimensions
    (all dimensions except the last ``2``).

    For example:

    * ``x.shape == (10, 4, 3)``
    * ``y.shape == (5, 10, 3, 2)``
    * ``matmul(x, y).shape == (5, 10, 4, 2)``

    Conventional matrix multiplication is a special case where both ``x, y`` are
    exactly 2-D. For example:

    * ``x.shape == (4, 3)``
    * ``y.shape == (3, 2)``
    * ``matmul(x, y).shape == (4, 2)``

    If ``x`` is 1-D, and ``y`` is N-D where ``N >= 2``, ``x`` is first promoted to
    matrix ``xm`` by prepending a ``1`` to its dimension, and the resulting ``xm`` is
    broadcast to ``y`` following Rule 2 above. After this, remove the inserted dimension.
    For example:

    * ``x.shape == (4)``
    * ``y.shape == (10, 4, 3)``
    * ``xm.shape == (1, 4)``
    * ``matmul(xm, y).shape == (10, 1, 3)``
    * Removing the inserted dimension results in ``matmul(x, y).shape == (10, 3)``.
    * Note: ``xm`` and ``matmul(xm, y)`` are for illustration only.

    If ``x`` is N-D where ``N >= 2``, and ``y`` is 1-D, ``y`` is first promoted to
    matrix ``ym`` by appending a ``1`` to its dimension, and the resulting ``ym`` is
    broadcast to ``x`` following Rule 2 above. After this, remove the inserted dimension.
    For example:

    * ``x.shape == (10, 3, 4)``
    * ``y.shape == (4,)``
    * ``ym.shape == (4, 1)``
    * ``matmul(x, ym).shape == (10, 3, 1)``
    * Removing the inserted dimension results in ``matmul(x, y).shape == (10, 3)``.
    * Note: ``xm`` and ``matmul(xm, y)`` are for illustration only.

    Parameters
    ----------
    x: tensor<[\*,K1], T> (Required)
        * ``x`` must be 1-D or higher.
    y: tensor<[\*,K2], T> (Required)
        * ``y`` must be 1-D or higher.
    transpose_x: const bool (Optional)
        * Default to ``False``.
        * Use ``True`` to transpose the last two dimensions of ``x`` before multiplication.
          It has no effect when ``x`` is 1-D.
    transpose_y: const bool (Optional)
        * Default to ``False``.
        * Use ``True`` to transpose the last two dimensions of ``y`` before multiplication.
          It has no effect when ``y`` is 1-D.

    Returns
    -------
    tensor<\*, T>
        * Scalar or tensor output.

    Attributes
    ----------
    T: fp32
    """
    input_spec = InputSpec(
        x=TensorInputType(),
        y=TensorInputType(),
        transpose_x=BoolInputType(const=True, optional=True),
        transpose_y=BoolInputType(const=True, optional=True),
    )


    def default_inputs(self):
        return DefaultInputs(
            transpose_x=False,
            transpose_y=False,
            )

    def __init__(self, **kwargs):
        super(matmul, self).__init__(**kwargs)

    def type_inference(self):
        x_type = self.x.dtype
        x_shape = list(self.x.shape)
        y_shape = list(self.y.shape)
        x_rank = len(x_shape)

        if x_rank == 1 and self.transpose_x.val:
            msg = "Op {} (matmul): x is rank 1, but transpose_x is True, which is not allowed."
            raise ValueError(msg.format(self.name))

        if self.transpose_x.val:
            x_shape = list(x_shape)
            x_shape[-1], x_shape[-2] = x_shape[-2], x_shape[-1]
            x_shape = tuple(x_shape)
        if self.transpose_y.val:
            y_shape = list(y_shape)
            y_shape[-1], y_shape[-2] = y_shape[-2], y_shape[-1]
            y_shape = tuple(y_shape)
        if not (
            x_shape[-1] == y_shape[-2]
            or is_symbolic(x_shape[-1])
            or is_symbolic(y_shape[-2])
        ):
            msg = "Op {} (matmul): x {}, y {} are not broadcastable"
            raise ValueError(msg.format(self.name, self.x.shape, self.y.shape))

        if x_rank == 1:
            # promote shape of x to rank 2
            x_shape = list((1,) + tuple(x_shape))
        ret_shape = list(broadcast_shapes(x_shape[:-2], y_shape[:-2]))
        ret_shape += [x_shape[-2], y_shape[-1]]
        if x_rank == 1:
            # remove the first dimension of the returned shape
            return types.tensor(x_type, tuple(ret_shape[1:]))
        else:
            return types.tensor(x_type, tuple(ret_shape))

    @precondition(allow=VALUE)
    def value_inference(self):
        x = self.x.val
        if self.transpose_x.val:
            x = np.transpose(x)
        y = self.y.val
        if self.transpose_y.val:
            y = np.transpose(y)
        return np.matmul(x, y)


@register_op(doc_str="")
class einsum(Operation):
    """
    Perform tensor multiplication expressed according to the einsum notation.
    The mode/equation that is currently supported is mutiplying matrices that are laid out on
    dimensions -1 and -3, treating all the other dimensions as batch. Broadcasting is supported along batch dimensions.
    In particular, the inputs must be of the following shapes:

    * rank 4 input case
        * input 1: [B, C, H, W1]
        * input 2: [B, W1, H, W2]
        * output: [B, C, H, W2]
        * if , for one of the inputs, the dimensions "B" or "H" is 1, they are broadcast to match the other input

    * rank 3 input case
        * input 1: [C, H, W1]
        * input 2: [W1, H, W2]
        * output: [C, H, W2]
        * if , for one of the inputs, the dimension "H" is 1, it is broadcast to match the other input

    Parameters
    ----------
    values : Tuple(tensor_1, tensor_2)
        * where
            * tensor_1: tensor<[*D, C, H, W1], T>
            * must be of rank 3 or 4
            * tensor_2: tensor<[*D, W1, H, W2], T>
            * must be of rank 3 or 4
    equation: const<str>
        * supported equations are:
            * "nchw,nwhu->nchu" and its equivalent equation strings
            * "chw,whr->chr" and its equivalent equation strings

    Returns
    -------
    tensor<[*D, C, H, W2], T>
        * same ranks as the inputs

    Attributes
    ----------
    T: fp32
    """

    input_spec = InputSpec(values=TupleInputType(),
                           equation=StringInputType(const=True))

    def __init__(self, **kwargs):
        super(einsum, self).__init__(**kwargs)

    def type_inference(self):
        if len(self.values) != 2:
            raise ValueError("einsum op must get \'values\' of length 2")
        x = self.values[0]
        y = self.values[1]

        # validate the input shapes
        x_type = x.dtype
        assert x_type == y.dtype, "input types do not match"
        x_shape = x.shape
        y_shape = y.shape
        assert len(x_shape) == len(y_shape), "inputs not of the same rank"
        assert x_shape[-1] == y_shape[-3], "input shapes incompatible"
        if x_shape[-2] != 1 and y_shape[-2] != 1:
            assert x_shape[-2] == y_shape[-2], "input shapes incompatible"
        if len(x_shape) == 4:
            if x_shape[-4] != 1 and y_shape[-4] != 1:
                assert x_shape[-4] == y_shape[-4], "input shapes incompatible"

        # validate the equation
        input1_vec, input2_vec, output_vec = parse_einsum_equation(self.equation.val)

        assert \
            (input1_vec == [0, 1, 2, 3] and input2_vec == [0, 3, 2, 4] and output_vec == [0, 1, 2, 4]) or \
            (input1_vec == [0, 1, 2] and input2_vec == [2, 1, 3] and output_vec == [0, 1, 3]), \
            "unsupported einsum equation {}".format(self.equation.val)

        # calculate the output shape
        def _get_dim_value(shape1, shape2, dim):
            if is_symbolic(shape1[dim]) and is_symbolic(shape2[dim]):
                return shape1[dim]
            elif is_symbolic(shape1[dim]):
                return shape1[dim]
            elif is_symbolic(shape2[dim]):
                return shape2[dim]
            else:
                return max(shape1[dim], shape2[dim])

        out_shape = [1 for i in range(len(x_shape))]
        out_shape[-1] = y_shape[-1]
        out_shape[-3] = x_shape[-3]
        out_shape[-2] = _get_dim_value(x_shape, y_shape, -2)
        if len(x_shape) == 4:
            out_shape[-4] = _get_dim_value(x_shape, y_shape, -4)
        return types.tensor(x_type, tuple(out_shape))

    @precondition(allow=VALUE)
    def value_inference(self):
        x = self.values[0]
        y = self.values[1]
        x_shape = x.val.shape
        y_shape = y.val.shape
        # broadcast dimensions -2 and -4, if required
        if len(x_shape) == 4:
            x_shape = (max(x_shape[0], y_shape[0]), x_shape[1], max(x_shape[2], y_shape[2]), x_shape[3])
            y_shape = (max(x_shape[0], y_shape[0]), y_shape[1], max(x_shape[2], y_shape[2]), y_shape[3])
        elif len(x_shape) == 3:
            x_shape = (x_shape[0], max(x_shape[1], y_shape[1]), x_shape[2])
            y_shape = (y_shape[0], max(x_shape[1], y_shape[1]), y_shape[2])
        else:
            raise ValueError("ranks of the input must be 3 or 4")
        res = np.einsum(self.equation.val,
                        np.broadcast_to(x.val, x_shape),
                        np.broadcast_to(y.val, y_shape))
        return res
