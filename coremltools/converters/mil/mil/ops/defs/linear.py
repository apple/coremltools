#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from coremltools.converters.mil.mil.types.symbolic import is_symbolic
from ._op_reqs import *
from ._utils import broadcast_shapes


@register_op(
    doc_str="""
Performs  x*weight.T + bias where weight and bias are const at compile time.

Inputs

* x: <*D,D_in,T> (Required)
    * 1 <= rank <= 3
    * 0 <= rank(*D) <= 2
* weight: const<D_out,D_in,T> (Required)
* bias: const<D_out,T> (Optional. Default to 0)

Outputs

* <*D,D_out,T>
    * same rank as the input

Type Domains

* T: f32
"""
)
class linear(Operation):
    input_spec = InputSpec(
        x=TensorInputType(),
        weight=TensorInputType(const=True),
        bias=TensorInputType(const=True, optional=True),
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


# rdar://58622145
@register_op(doc_str="TODO")
class matmul(Operation):
    input_spec = InputSpec(
        x=TensorInputType(),
        y=TensorInputType(),
        transpose_x=BoolInputType(const=True, default=False),
        transpose_y=BoolInputType(const=True, default=False),
    )

    def __init__(self, **kwargs):
        super(matmul, self).__init__(**kwargs)

    def type_inference(self):
        # rdar://58621799 TODO: handle 1D x, y
        x_type = self.x.dtype
        x_shape = list(self.x.shape)
        y_shape = list(self.y.shape)

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

        ret_shape = list(broadcast_shapes(x_shape[:-2], y_shape[:-2]))
        ret_shape += [x_shape[-2], y_shape[-1]]
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
