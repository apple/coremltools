#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from coremltools.converters.mil.mil import (SYMBOL, VALUE)
from coremltools.converters.mil.mil.types.symbolic import is_compatible_symbolic_vector
from ._op_reqs import *
import numbers

@register_op(doc_str="TODO")
class gather(Operation):
    input_spec = InputSpec(
            x = TensorInputType(),
            indices = IntOrIntTensorInputType(),
            axis = IntInputType(const=True, default=0)
            )

    def __init__(self, **kwargs):
        super(gather, self).__init__(**kwargs)

    @precondition(allow=VALUE|SYMBOL)
    def value_inference(self):
        x = self.x.sym_val
        indices = self.indices.val
        if indices is None:
            # only allow x to be symbolic. indices cannot.
            return None
        scalar_indices = isinstance(indices, numbers.Integral)
        axis = self.axis.val
        if scalar_indices:
            res = np.take(x, [indices], axis)
            res2 = np.squeeze(res, axis=axis)
            if isinstance(res2, np.ndarray) and len(res2.shape) == 0:
                # res2 is a scalar, but represented as np.array(symbol,
                # dtype=np.object) which np.squeeze can't remove.
                return res2.item()
            return res2
        return np.take(x, indices, axis)

    def type_inference(self):
        out_type = self.x.dtype

        if self.axis.val < -self.x.rank \
                or self.axis.val >= self.x.rank:
            raise IndexError(
                'Axis value {} is out of bounds for {} node {}'.format(
                    self.axis.val, self.op_type, self.name))

        output_rank = self.x.rank - 1 + self.indices.rank
        if output_rank == 0:
            # output scalar
            return out_type

        axis = self.axis.val
        axis = axis if axis >= 0 else axis + self.x.rank
        out_shape = self.x.shape[:axis] + self.indices.shape + self.x.shape[axis + 1:]
        return types.tensor(out_type, out_shape)


@register_op(doc_str="TODO")
class scatter(Operation):
    input_spec = InputSpec(
            data = TensorInputType(),
            indices = IntTensorInputType(),
            updates = TensorInputType(),
            axis = IntInputType(const=True, default=0),
            mode = StringInputType(const=True, default="add")
            )

    def __init__(self, **kwargs):
        super(scatter, self).__init__(**kwargs)

    def type_inference(self):
        if self.axis.val < -self.data.rank \
                or self.axis.val >= self.data.rank:
            raise IndexError(
                'Axis value {} is out of bounds for {} node {}'.format(
                    self.axis.val, self.op_type, self.name))

        axis = self.axis.val
        axis = axis if axis >= 0 else axis + self.data.rank
        expected_updates_shape = self.data.shape[:axis] + self.indices.shape + self.data.shape[axis + 1:]
        np.testing.assert_equal(self.updates.shape, np.array(expected_updates_shape))

        return self.data.sym_type


@register_op(doc_str="TODO")
class gather_along_axis(Operation):
    input_spec = InputSpec(
            x = TensorInputType(),
            indices = IntTensorInputType(),
            axis = IntInputType(const=True, default=0)
            )

    def __init__(self, **kwargs):
        super(gather_along_axis, self).__init__(**kwargs)

    @precondition(allow=VALUE)
    def value_inference(self):
        x = self.x.val
        indices = self.indices.val
        axis = self.axis.val
        return np.take_along_axis(x, indices, axis)

    def type_inference(self):

        if self.x.rank != self.indices.rank:
            raise ValueError("Rank mismatch between input and indices. \
                              Input rank: {}, indices rank: {}".format(self.x.rank, self.indices.rank))

        if self.axis.val < -self.x.rank \
                or self.axis.val >= self.x.rank:
            raise IndexError(
                'Axis value {} is out of bounds for {} node {}'.format(
                    self.axis.val, self.op_type, self.name))

        axis = self.axis.val
        axis = axis if axis >= 0 else axis + self.x.rank

        for i in range(self.x.rank):
            if i != axis:
                assert self.x.shape[i] == self.indices.shape[i]

        return types.tensor(self.x.dtype, self.indices.shape)


@register_op(doc_str="TODO")
class scatter_along_axis(Operation):
    input_spec = InputSpec(
            data = TensorInputType(),
            indices = IntTensorInputType(),
            updates = TensorInputType(),
            axis = IntInputType(const=True, default=0),
            mode = StringInputType(const=True, default="add")
            )

    def __init__(self, **kwargs):
        super(scatter_along_axis, self).__init__(**kwargs)

    @precondition(allow=VALUE)
    def value_inference(self):
        data = np.copy(self.data.val)
        indices = self.indices.val
        updates = self.updates.val
        axis = self.axis.val
        np_output = data
        np.put_along_axis(np_output, indices, updates, axis=axis)
        return np_output

    def type_inference(self):
        if self.axis.val < -self.data.rank \
                or self.axis.val >= self.data.rank:
            raise IndexError(
                'Axis value {} is out of bounds for {} node {}'.format(
                    self.axis.val, self.op_type, self.name))

        axis = self.axis.val
        axis = axis if axis >= 0 else axis + self.data.rank

        assert self.indices.shape == self.updates.shape
        assert self.data.rank == self.indices.rank
        for i in range(self.data.rank):
            if i != axis:
                assert self.data.shape[i] == self.indices.shape[i]

        return self.data.sym_type


@register_op(doc_str="TODO")
class gather_nd(Operation):
    input_spec = InputSpec(
            x = TensorInputType(),
            indices = IntTensorInputType(),
            )

    def __init__(self, **kwargs):
        super(gather_nd, self).__init__(**kwargs)

    def type_inference(self):
        assert self.indices.shape[-1] <= self.x.rank
        out_type = self.x.dtype
        out_shape = self.indices.shape[:-1] + self.x.shape[self.indices.shape[-1]:]
        return types.tensor(out_type, out_shape)


@register_op(doc_str="TODO")
class scatter_nd(Operation):
    input_spec = InputSpec(
            data = TensorInputType(),
            indices = IntTensorInputType(),
            updates = TensorInputType(),
            mode = StringInputType(const=True, default="add")
            )

    def __init__(self, **kwargs):
        super(scatter_nd, self).__init__(**kwargs)

    def type_inference(self):
        assert self.indices.shape[-1] <= self.data.rank
        expected_updates_shape = self.indices.shape[:-1] + self.data.shape[self.indices.shape[-1]:]
        assert is_compatible_symbolic_vector(self.updates.shape, tuple(expected_updates_shape))
        return self.data.sym_type
