from __future__ import division

from coremltools.converters.nnv2.nnv2_program.program.program import get_new_symbol
from ._op_reqs import *

@register_op(doc_str='TODO')
class slice_by_index(Operation):
    input_spec = InputSpec(
        x=TensorInputType(),
        begin=IntTensorInputType(),
        end=IntTensorInputType(),
        stride=IntTensorInputType(const=True, optional=True),
        begin_mask=BoolTensorInputType(const=True, optional=True),
        end_mask=BoolTensorInputType(const=True, optional=True),
        squeeze_mask=BoolTensorInputType(const=True, optional=True)
    )

    def __init__(self, **kwargs):
        super(slice_by_index, self).__init__(**kwargs)

    def type_inference(self):
        if self.begin.rank != 1:
            raise ValueError("begin should be 1-D tensor, got {}-D tensor instead".format(self.begin.rank))
        if self.end.rank != 1:
            raise ValueError("end should be 1-D tensor, got {}-D tensor instead".format(self.end.rank))
        if self.x.rank != self.begin.shape[0]:
            raise ValueError("Length of begin {} doesn't equal to input rank {}.".format(len(self.begin.shape[0]), len(self.x.rank)))
        if self.x.rank != self.end.shape[0]:
            raise ValueError("Length of end {} doesn't equal to input rank {}.".format(len(self.end.shape[0]), len(self.x.rank)))

        x_shape = self.x.shape
        ret_shape = []
        rank_offset = 0
        if self.squeeze_mask is not None:
            squeeze_mask = self.squeeze_mask.val
        else:
            squeeze_mask = [False for _ in range(self.x.rank)]
        for s in squeeze_mask:
            if s:
                rank_offset += 1

        if self.begin.sym_val is None and self.begin_mask is None:
            ret_shape = [get_new_symbol() for _ in range(len(x_shape)-rank_offset)]
            return builtins.tensor(self.x.dtype, tuple(ret_shape))
        if self.end.sym_val is None and self.end_mask is None:
            ret_shape = [get_new_symbol() for _ in range(len(x_shape)-rank_offset)]
            return builtins.tensor(self.x.dtype, tuple(ret_shape))

        begin = self.begin.sym_val
        end = self.end.sym_val
        stride = [1] * self.x.rank if self.stride is None else self.stride.val
        if begin is None:
            begin_mask = self.begin_mask.val
            begin = [None for _ in range(len(x_shape))]
            for idx, mask in enumerate(begin_mask):
                if mask:
                    begin[idx] = 0 if stride[idx] > 0 else x_shape[idx]-1
        if end is None:
            end_mask = self.end_mask.val
            end = [None for _ in range(len(x_shape))]
            for idx, mask in enumerate(end_mask):
                if mask:
                    # This is not totally correct, we need to read end_mask to take care of this case
                    end[idx] = x_shape[idx] if stride[idx] > 0 else 0

        if None in end or None in begin:
            ret_shape = []
            for idx in range(len(x_shape)):
                if squeeze_mask[idx]:
                    continue
                if begin[idx] is not None and end[idx] is not None:
                    if stride[idx] > 0:
                        ret_shape.append(np.ceil((end[idx]-begin[idx])/stride[idx]))
                    elif end_mask[idx]:
                        ret_shape.append(np.ceil((-1-begin[idx])/stride[idx]))
                    else:
                        ret_shape.append(np.ceil((end[idx]-begin[idx])/stride[idx]))
                else:
                    ret_shape.append(get_new_symbol())
            return builtins.tensor(self.x.dtype, tuple(ret_shape))

        for idx, b in enumerate(begin):
            if b >= 0:
                continue
            begin[idx] += x_shape[idx]
        for idx, e in enumerate(end):
            if e >= 0:
                continue
            end[idx] += x_shape[idx]
        stride = [1 for _ in range(len(x_shape))] if self.stride is None else self.stride.val
        end_mask = [False] * len(x_shape) if self.end_mask is None else self.end_mask.val
        squeeze_mask = [False] * len(x_shape) if self.squeeze_mask is None else self.squeeze_mask.val
        for idx in range(len(x_shape)):
            if squeeze_mask[idx]:
                continue
            if end_mask[idx] and stride[idx] < 0:
                ret_shape.append(np.ceil((-1-begin[idx])/stride[idx]).astype(np.int32))
            else:
                ret_shape.append(np.ceil((end[idx]-begin[idx])/stride[idx]).astype(np.int32))

        if len(ret_shape) == 0:
            # Scalar case.
            return self.x.dtype
        else:
            return builtins.tensor(self.x.dtype, tuple(ret_shape))

    def value_inference(self):
        if self.x.sym_val is None or self.begin.val is None or self.end.val is None:
            return None
        x_shape = self.x.shape
        begin = [int(i) for i in list(self.begin.val[:])]
        end = [int(i) for i in list(self.end.val[:])]
        stride = [1] * self.x.rank if self.stride is None else self.stride.val
        begin_mask = [False] * self.x.rank if self.begin_mask is None else self.begin_mask.val
        end_mask = [False] * self.x.rank if self.end_mask is None else self.end_mask.val
        squeeze_mask = [False] * self.x.rank if self.squeeze_mask is None else self.squeeze_mask.val

        slices = []
        for idx, mask in enumerate(begin_mask):
            if mask:
                begin[idx] = None
        for idx, mask in enumerate(end_mask):
            if mask:
                end[idx] = None
        squeeze_axes = []
        for idx, mask in enumerate(squeeze_mask):
            if mask:
                end[idx] = None
                stride[idx] = 2147483647 # We slice out only 1 element by setting stride to INF
                squeeze_axes.append(idx)
        for idx in range(self.x.rank):
            slices.append(slice(begin[idx], end[idx], stride[idx]))

        slices = tuple(slices)
        res = self.x.sym_val[slices]

        # remove squeezed axes
        if len(squeeze_axes) > 0:
            if len(squeeze_axes) == len(res.shape):
                if len(res) == 0:
                    logging.warning("%s seems to be a 0 sized tensor", self.name)
                    return np.array([])
                res = res.tolist()[0]
                if self.x.sym_val.dtype == np.int32 or self.x.sym_val.dtype == np.int64:
                    res = np.int32(res)
                elif self.x.sym_val.dtype == np.float32 or self.x.sym_val.dtype == np.float64:
                    res = np.float32(res)
                else:
                    raise ValueError("Unable to convert type {}".format(self.x.sym_val.dtype))
            else:
                res = np.squeeze(res, axis=tuple(squeeze_axes))
        return res

