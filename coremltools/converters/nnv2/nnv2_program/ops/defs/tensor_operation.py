import math
from coremltools.converters.nnv2.builtin_types.symbolic import is_symbolic, any_symbolic
from coremltools.converters.nnv2.nnv2_program.program.program import (
        get_new_symbol, get_new_variadic_symbol, SYMBOL, VALUE, NONE)
from ._op_reqs import *
from ._utils import _promoted_primitive_type

@register_op(doc_str="""
Returns a tensor setting everything outside a center band to zeros for the innermost matrix. Special cases:

* band_part(x, 0, -1) returns upper triangular part.
* band_part(x, -1, 0) returns lower triangular part.
* band_part(x, 0, 0) returns diagonal.

Inputs

* x <*, T> Required
    * Input tensor.
* lower: const<i32> Optional
    * Number of lower / below sub-diagonals to keep. If negative, keep entire lower triangle.
    * Defaults to -1 (keep the entire lower triangle)
* upper: const<i32> Optional
    * Number of upper / above sub-diagonals to keep. If negative, keep entire lower triangle.
    * Defaults to -1 (keep the entire upper triangle)

Outputs

* <*, T> same type as the input tensor.

Type Domains

* T: f32
""")
class band_part(Operation):
    input_spec = InputSpec(
        x=TensorInputType(),
        lower=IntInputType(const=True, default=-1),
        upper=IntInputType(const=True, default=-1),
    )

    def __init__(self, **kwargs):
        super(band_part, self).__init__(**kwargs)

    def type_inference(self):
        return self.x.sym_type


@register_op(doc_str='TODO')
class cumsum(Operation):
    input_spec = InputSpec(
            x = TensorInputType(),
            axis = IntInputType(const=True, default=0),
            exclusive = BoolInputType(const=True, default=False),
            reverse = BoolInputType(const=True, default=False)
            )

    def __init__(self, **kwargs):
        super(cumsum, self).__init__(**kwargs)

    @precondition(allow=VALUE)
    def value_inference(self):
        data = np.copy(self.x.val)
        axis = self.axis.val
        reverse = self.reverse.val
        exclusive = self.exclusive.val
        if reverse:
            data = np.flip(data, axis=axis)
        data = np.cumsum(data, axis=axis)
        if exclusive:
            zero_shape = np.copy(data.shape)
            zero_shape[axis] = 1
            data = np.concatenate((np.zeros(zero_shape, data)), axis=axis)
        if reverse:
            data = np.flip(data, axis=axis)
        return data

    def type_inference(self):
        #Check range of axis
        if self.axis.val < -1 or self.axis.val > self.x.rank - 1:
            raise ValueError("axis should be in the range [-1, {}]".format(self.x.rank - 1))

        return self.x.sym_type


@register_op(doc_str="""
Returns a tensor with given shape filled with a constant value.

Parameters
----------
shape: <K, i32>, required
    Target output tensor shape.
    K is the rank of the output tensor. shape[k] > 0 for k = 0,..., K-1.
value: const<f32>, optional
    Constant value to fill in. Defaults to 0.

Returns
-------
A tensor.
""")
class fill(Operation):
    input_spec = InputSpec(
        shape=IntTensorInputType(),
        value=FloatInputType(const=True, default=0.),
    )

    def __init__(self, **kwargs):
        super(fill, self).__init__(**kwargs)

    def type_inference(self):
        if any_symbolic(self.shape.shape):
            # We can't infer any shape if shape has variable length.
            return builtins.tensor(builtins.fp32, (get_new_variadic_symbol(),))

        # shape has fixed length here.
        if self.shape.sym_val is None:
            shape = tuple([get_new_symbol() for _ in range(self.shape.shape[0])])
            return builtins.tensor(builtins.fp32, shape)

        return builtins.tensor(builtins.fp32, tuple(self.shape.sym_val.tolist()))

    @precondition(allow=VALUE)
    def value_inference(self):
        return np.full(shape=self.shape.val, fill_value=self.value.val)


@register_op(doc_str="""
Applies non-maximum suppression (NMS) on the input box coordinates according
to their intersection-over-union (IoU). NMS selects as subset of bounding
boxes with the descending scores. Removes boxes that have high
intersection-over-union (IOU) overlap with previously selected boxes.

Input

* boxes: <n, B, 4, f32>
    * Box coordinates to perform NMS on.
* scores: <n, B, K, f32>
    * Scores for each one of the boxes
* iou_threshold: const<f32>
    * The intersection over union (IoU) threshold over which boxes are suppressed. NMS remove all overlapping boxes with IoU > iou_threshold
* score_threshold: const<f32>
    * Before IoU suppression is performed, boxes with class scores below this threshold are rejected
* max_boxes: const<i32>
    * Maximum number of boxes to select. If the number of surviving boxes are less, output is padded up to this number
* per_class_suppression: const<bool>
    * Optional, defaults to False
    * If true, suppression is performed independently within boxes of each class

Output

* <n, max_boxes, 4, f32>
    * Coordinates of selected boxes
* <n, max_boxes, K, f32>
    * Scores of selected boxes
* <n, max_boxes, i32>
    * Indices of selected boxes
* <n, i32>
    * Number of boxes selected for each batch
""")
class non_maximum_suppression(Operation):
    input_spec = InputSpec(
        boxes=TensorInputType(),
        scores=TensorInputType(),
        iou_threshold=FloatInputType(const=True),
        score_threshold=FloatInputType(const=True),
        max_boxes=IntInputType(const=True),
        per_class_suppression=BoolInputType(const=True, default=False)
    )

    def __init__(self, **kwargs):
        super(non_maximum_suppression, self).__init__(**kwargs)

    def type_inference(self):
        boxes_dtype = self.boxes.dtype
        scores_dtype = self.scores.dtype
        n_batch, _, n_score = self.scores.shape
        max_boxes = self.max_boxes.val

        return builtins.tensor(boxes_dtype, (n_batch, max_boxes, 4)), \
               builtins.tensor(scores_dtype, (n_batch, max_boxes, n_score)), \
               builtins.tensor(builtins.int32, (n_batch, max_boxes)), \
               builtins.tensor(builtins.int32, (n_batch,))


@register_op(doc_str="""
Returns the indices of the elements in the given tensor that are non-zero.

Inputs

* x <*, T>
    * Tensor, values selected at indices where its values is not equal to 0.

Outputs

* <N, R, T>
    * 2-dimensional tensor contains indices of elements that are non-zero. Each
    row is the index for a non-zero value. N is the number of non-zero elements,
    R is the rank of the input.

Type Domains

* T: f32
""")
class non_zero(Operation):
    input_spec = InputSpec(
        x=TensorInputType()
    )

    def __init__(self, **kwargs):
        super(non_zero, self).__init__(**kwargs)

    def type_inference(self):
        shape = tuple([get_new_symbol(), self.x.rank])
        return builtins.tensor(self.x.dtype, shape)

    @precondition(allow=VALUE)
    def value_inference(self):
        return np.transpose(np.nonzero(self.x.val))


# rdar://59195036
@register_op(doc_str="TODO")
class one_hot(Operation):
    input_spec = InputSpec(
            indices = IntTensorInputType(),
            one_hot_vector_size=IntInputType(),
            axis  = IntInputType(const=True, default=-1),
            on_value  = IntOrFloatInputType(const=True, default=1),
            off_value = IntOrFloatInputType(const=True, default=0)
    )

    def __init__(self, **kwargs):
        super(one_hot, self).__init__(**kwargs)

    def type_inference(self):
        on_type = self.on_value.dtype
        off_type = self.off_value.dtype

        if on_type != off_type:
            raise TypeError("Parameters on_value and off_value must have same input types.")

        if self.axis.val < -self.indices.rank - 1 \
                or self.axis.val > self.indices.rank:
            raise IndexError(
                'Axis value {} is out of bounds for {} node {}'.format(
                    self.axis.val, self.op_type, self.name))

        indices_shape = list(self.indices.shape)

        depth_value = self.one_hot_vector_size.sym_val
        if depth_value is None:
            depth_value = get_new_symbol()
        elif depth_value < 0:
            raise ValueError('Parameter one_hot_vector_size must be non-negative')

        retshape = indices_shape

        if self.axis.val < 0:
            cut = len(retshape) + self.axis.val + 1
        else:
            cut = self.axis.val
        retshape = retshape[0:cut] + [depth_value] + retshape[cut:]

        return builtins.tensor(on_type, retshape)


# rdar://58622145
@register_op(doc_str='TODO')
class pad(Operation):
    input_spec = InputSpec(
            x = TensorInputType(),
            pad = IntTensorInputType(const=True),
            mode = StringInputType(const=True, default="constant"),
            constant_val = FloatInputType(const=True, default=0.),
            )

    def __init__(self, **kwargs):
        super(pad, self).__init__(**kwargs)

    def type_inference(self):
        in_shape = self.x.shape
        pad = self.pad.val
        ret_shape = list(in_shape)
        pad_index = 0

        if pad.shape[0] % 2 != 0:
            raise ValueError("Padding must be even! Provided {}".format(self.pad.shape[0]))

        mode = self.mode.val
        if mode in {"reflect", "replicate"}:
            # Reflect and Replicate mode only support padding to last two dimensions
            if self.pad.shape[0] != 4:
                msg = "For {} mode, padding is only supported on last two dimension, provided" + \
                      " {}".format(mode, self.pad.val)
                raise ValueError("Incorrect Pad configuration! {}".format(msg))

        for i in range(len(ret_shape) - pad.shape[0] // 2, len(ret_shape)):
            ret_shape[i] = in_shape[i] + pad[2*pad_index] + pad[2*pad_index + 1]
            pad_index += 1
        return builtins.tensor(self.x.dtype, ret_shape)

    @precondition(allow=VALUE)
    def value_inference(self):
        # NumPy `edge` mode is equivalent to `replicate` mode of PyTorch and CoreML
        mode = 'edge' if self.mode.val == 'replicate' else self.mode.val
        pad_val = self.pad.val
        if len(self.x.val.shape) > (pad_val.shape[0] // 2):
            updated_pad = np.zeros(len(self.x.val.shape)*2)
            updated_pad[-pad_val.shape[0]:] = pad_val
            pad_val = updated_pad
        pad_val = pad_val.reshape(-1, 2).astype(np.int32)
        if mode == 'constant':
            return np.pad(self.x.val, pad_val, mode,
                          constant_values=self.constant_val.val)
        # NumPy does not support non-constant mode and constant_values argument
        return np.pad(self.x.val, pad_val, mode)


# rdar://59195036
@register_op(doc_str="TODO")
class range_1d(Operation):
    input_spec = InputSpec(
            end   = IntOrFloatInputType(),
            start = IntOrFloatInputType(),
            step  = IntOrFloatInputType()
            )

    def __init__(self, **kwargs):
        super(range_1d, self).__init__(**kwargs)

    @precondition(allow=VALUE)
    def value_inference(self):
        start = self.start.val
        end   = self.end.val
        step  = self.step.val
        return np.arange(start, end, step)

    def type_inference(self):
        start = self.start.sym_val
        end = self.end.sym_val
        step = self.step.sym_val

        if (self.start.dtype != self.end.dtype) or (self.start.dtype != self.step.dtype) or (
            self.end.dtype != self.step.dtype):
            raise TypeError("All inputs to the range operation must have same input types.")

        if all(sym_val is not None for sym_val in (start, end, step)):
            shape = (end - start) / step
            shape = shape if is_symbolic(shape) else int(math.ceil(shape))
            shape = tuple([shape])
        else:
            shape = tuple([get_new_symbol(),])

        return builtins.tensor(self.start.dtype, shape)


@register_op(doc_str='TODO')
class tile(Operation):
    input_spec = InputSpec(
            x = TensorInputType(),
            reps = IntTensorInputType(const=True),
            )

    def __init__(self, **kwargs):
        super(tile, self).__init__(**kwargs)

    def type_inference(self):
        x_type = self.x.dtype
        x_shape = np.array(self.x.shape)
        reps = self.reps.val
        if len(reps) == 0  or len(reps) > self.x.rank:
            raise ValueError("Length of the reps parameter must be at least 1 and " \
                                                 "not greater than the rank of the input, x")

        if any(i <=0 for i in reps):
            raise ValueError("All entries of reps paramerer must be greater than 0")

        if len(reps) < self.x.rank:
            reps = [1]*(self.x.rank - len(reps)) + list(reps)

        out_shape =  tuple([reps[i] * x_shape[i] for i in range(len(reps))])

        return builtins.tensor(x_type, out_shape)

    @precondition(allow=VALUE)
    def value_inference(self):
        return np.tile(self.x.val, reps=self.reps.val)


@register_op(doc_str="""
Returns a tensor containing the indices of the sorted values along given axis
of the input tensor.

Inputs

* x: <*, T> Required
    * Input tensor.
* axis: const<i32> or const<D, i32> Optional
    * Axis to perform the operation.
* ascending: const<bool> Optional
    * True to sort in ascending order. Default to false, sort in descending order

Outputs

* <*, T>

Type Domains

* T: f32
""")
class argsort(Operation):
    input_spec = InputSpec(
        x=TensorInputType(),
        axis=IntInputType(const=True, default=-1),
        ascending=BoolInputType(const=True, default=False)
    )

    def __init__(self, **kwargs):
        super(argsort, self).__init__(**kwargs)

    def type_inference(self):
        return builtins.tensor(builtins.int32, self.x.shape)

    @precondition(allow=VALUE)
    def value_inference(self):
        if self.ascending.val:
            return np.argsort(-self.x.val, axis=self.axis.val)
        return np.argsort(self.x.val, axis=self.axis.val)


@register_op(doc_str="""
Returns a tensor containing top or bottom k values and the corresponding
indices of the input tensor along a given axis.

Inputs

* x: <*, T> Required
    * Input tensor.
* k: const<i32>  (Optional. Default to 1)
    * Number of values/indices to be computed along each axis.
* axis: const<i32> Optional
    * Axis to perform the operation. Defaults to 1 (channel dimension).
* ascending: const<bool> Optional
    * Whether or not to sort in ascending order, defaults to false, sort in descending order.

Outputs

* <*, T>
    * Values of top/bottom k elements
* <*, T>
    * Indices of the top/bottom k elements along axis

Type Domains

* T: f32
""")
class topk(Operation):
    input_spec = InputSpec(
        x=TensorInputType(),
        k=IntInputType(const=True, default=1),
        axis=IntInputType(const=True, default=-1),
        ascending=BoolInputType(const=True, default=False)
    )

    def __init__(self, **kwargs):
        super(topk, self).__init__(**kwargs)

    def type_inference(self):
        x_type = self.x.dtype
        x_shape = self.x.shape
        k = self.k.val
        axis = self.axis.val

        if not is_symbolic(x_shape[axis]) and k > x_shape[axis]:
            msg = 'K={} is greater than size of the given axis={}'
            raise ValueError(msg.format(k, axis))

        ret_shape = list(x_shape)
        ret_shape[axis] = k
        return builtins.tensor(x_type, ret_shape), builtins.tensor(builtins.int32, ret_shape)

    @precondition(allow=VALUE)
    def value_inference(self):
        indices = np.argsort(self.x.val, axis=self.axis.val)
        if not self.ascending.val:
            indices = np.argsort(-self.x.val, axis=self.axis.val)
        slc = [slice(None)] * self.x.rank
        slc[self.axis.val] = slice(0, self.k.val)
        indices = indices[tuple(slc)]
        values = np.take_along_axis(self.x.val, indices, axis=self.axis.val)
        return values, indices


@register_op(doc_str="""
Flattens input tensor into 2d tensor by flattening dimensions before and after the provided axis

Inputs

* x: <*d, T> (Required)
* axis: const<f32>  Optional, defaults to 1
        negative axis is supported.

Outputs

* <d_prior, d_post, T>
    * d_prior is product of dimensions x[:axis]
    * d_post is product of dimensions x[axis:]

Type Domains

* T: f32

Examples

    1. input_shape = (3, ), axis = -1, output_shape = (1, 3)
    2. input_shape = (3, ), axis = 1, output_shape = (3, 1)
    3. input_shape = (4, 3), axis = -1, output_shape = (4, 3)
    4. input_shape = (2, 3, 2), axis = -1, output_shape = (6, 2)
    5. input_shape = (5, 5, 2), axis = 1, output_shape = (2, 10)
""")
class flatten(Operation):
    input_spec = InputSpec(
              x = TensorInputType(),
           axis = IntInputType(const=True, default=1)
    )

    def __init__(self, **kwargs):
        super(flatten, self).__init__(**kwargs)

    def type_inference(self):
        shape = list(self.x.shape)
        axis = self.axis.val
        dim_pre_axis = np.prod(shape[:axis])
        dim_post_axis = np.prod(shape[axis:])
        new_shape = [dim_pre_axis, dim_post_axis]
        return builtins.tensor(self.x.dtype, tuple(new_shape))

    @precondition(allow=VALUE|SYMBOL)
    def value_inference(self):
        shape = self.x.shape
        axis = self.axis.val

        dim_pre_axis = np.prod(shape[:axis])
        dim_post_axis = np.prod(shape[axis:])
        return self.x.val.reshape(dim_pre_axis, dim_post_axis)


@register_op(doc_str="""
Returns 1-dimensional tensor with shape of input tensor

Inputs

* x: <*d_in, T> (Required)

Outputs

* <K, i32>
    * Shape of input tensor
    * K = x.rank = len(d_in)

Type Domains

* T: f32
""")
class shape(Operation):
    input_spec = InputSpec(
              x = TensorInputType()
    )

    def __init__(self, **kwargs):
        super(shape, self).__init__(**kwargs)

    def type_inference(self):
        # TODO: rdar://60250739 ([NNv2] Allow Variadic rank for get_shape type_inference)
        input_rank = self.x.rank
        return builtins.tensor(builtins.int32, tuple([input_rank]))

    def value_inference(self):
        return np.array(self.x.shape)


@register_op(doc_str='TODO')
class concat(Operation):
    input_spec = InputSpec(
        values = TupleInputType(),
        axis = IntInputType(const=True),
    )

    def __init__(self, **kwargs):
        super(concat, self).__init__(**kwargs)

    def type_inference(self):
        concat_dim_len = 0
        if len(self.values) == 0:
            raise ValueError('Concat {} got 0 values'.format(self.name))

        # Validate values have the same rank
        rank = self.values[0].rank
        for v in self.values:
            if not builtins.is_tensor(v.sym_type):
                msg = 'Concat encounter non-tensor input {} of type {}'
                raise ValueError(msg.format(v.name, v.sym_type))
            if v.rank != rank:
                msg = 'Input {} has rank {} != other inputs rank {}'
                raise ValueError(msg.format(v.name, v.rank, rank))

        # Check concat axis is within (-rank, rank)
        concat_axis = self.axis.val
        if concat_axis < 0:
            concat_axis += rank
        if concat_axis < 0 or concat_axis >= rank:
            msg = 'In {} of op_type {}: axis out of bound for input '+\
                '(rank {})'
            raise ValueError(msg.format(self.name, self.op_type, rank))

        # Validate primitive types are compatible 
        dtype = self.values[0].dtype
        for v in self.values[1:]:
            new_dtype = _promoted_primitive_type(v.dtype, dtype)
            if new_dtype is None:
                msg = 'Incompatible primitive types concat: {} vs {}'
                raise ValueError(msg.format(v.dtype, dtype))
            dtype = new_dtype

        # validate that non-axis dimensions match
        retshape = list(self.values[0].shape)
        for v in self.values[1:]:
            for i in range(rank):
                if i != concat_axis and retshape[i] != v.shape[i]:
                    msg = 'Dimension mismatch in {} op {}'
                    raise ValueError(msg.format(self.op_type, self.name))

        # Get length of concat dim
        concat_dim_len = 0
        for v in self.values:
            taxis = v.shape[concat_axis]
            if is_symbolic(taxis):
                concat_dim_len = Symbol(self.name + '_axis')
                break
            concat_dim_len += taxis

        retshape[concat_axis] = concat_dim_len

        return builtins.tensor(dtype, retshape)

    @precondition(allow=VALUE|SYMBOL|NONE)
    def value_inference(self):
        has_values = True
        for v in self.values:
            if v.sym_val is None:
                has_values = False
                break
        if has_values:
            values = [v.val for v in self.values]
            return np.concatenate(values, axis=self.axis.val)

        # If input shapes are small, we create symbolic values from symbols.
        ret_shape = self.type_inference().get_shape()
        num_elems = np.prod(ret_shape)
        if num_elems > 5:
            return None
        if any_symbolic(ret_shape):
            # Don't know the exact shape.
            return None
        arr = np.array([Symbol(self.name + '%d' % d) \
                for d in range(num_elems)])
        return arr.reshape(ret_shape)

@register_op(doc_str='TODO')
class split(Operation):
    input_spec = InputSpec(
        x = TensorInputType(),
        num_splits = IntInputType(const=True, optional=True),
        split_sizes = IntTensorInputType(const=True, optional=True),
        axis = IntInputType(const=True),
    )

    def __init__(self, **kwargs):
        super(split, self).__init__(**kwargs)

    def type_inference(self):
        num_splits, sizes = self._get_num_splits_and_sizes()
        x_shape = list(self.x.shape)
        ret_shapes = [x_shape[:] for _ in range(num_splits)]
        axis = self.axis.val
        for i, d in enumerate(sizes):
            ret_shapes[i][axis] = d
        return tuple([builtins.tensor(self.x.dtype, s) for s in ret_shapes])

    def _get_num_splits_and_sizes(self):
        """
        Return:
        - num_splits: int
        - sizes: list of int/symbols. Of length num_splits

        Raise ValueError if num_splits cannot be determined.
        """
        if self.num_splits is None and self.split_sizes is None:
            msg = 'At least one of num_splits and split_sizes '+\
                    'must be specified in split op {}'
            raise ValueError(msg.format(self.name))

        axis = self.axis.val

        if self.num_splits is not None:
            num_splits = self.num_splits.val
            if self.split_sizes is None:
                # Even split
                if self.x.shape[axis] % num_splits != 0:
                    msg = 'num_split {} does not divide split ' +\
                            'dim (length = {})'
                    raise ValueError(msg.format(num_splits,
                        self.x.shape[axis]))
                size = self.x.shape[axis] / num_splits
                return num_splits, [size]*num_splits

            # self.split_sizes is not None
            if self.split_sizes.sym_val is not None:
                return num_splits, self.split_sizes.sym_val

            # self.split_size.sym_val is None.
            sizes = [get_new_symbol() for _ in range(num_splits)]
            return num_splits, sizes

        # self.num_splits is None, self.split_sizes is not None
        if self.split_sizes.sym_val is not None:
            return len(self.split_sizes.sym_val), self.split_sizes.sym_val

        # self.num_splits is None, self.split_sizes is not None
        # self.split_sizes.sym_val is None
        if any_symbolic(self.split_sizes.shape):
            raise ValueError('Unable to determine number of splits')

        num_splits = len(self.split_sizes.shape)
        sizes = [get_new_symbol() for _ in range(num_splits)]
        return num_splits, sizes


    @precondition(allow=VALUE|SYMBOL|NONE)
    def value_inference(self):
        num_splits, sizes = self._get_num_splits_and_sizes()
        if self.x.sym_val is None or any_symbolic(sizes):
            raise NotImplementedError()

        if num_splits == 1:
            # No split_indices possible.
            return self.x.sym_val

        split_indices = np.cumsum(sizes).astype(np.int)
        return tuple(np.split(self.x.sym_val, split_indices[:-1]))

@register_op(doc_str='TODO')
class stack(Operation):
    input_spec = InputSpec(
        values = TupleInputType(),
        axis = IntInputType(const=True),
    )

    def __init__(self, **kwargs):
        super(stack, self).__init__(**kwargs)

    def type_inference(self):
        num_tensors = len(self.values)
        if num_tensors == 0:
            raise ValueError('Cannot stack 0 tensor')

        t_shape = self.values[0].shape
        for t in self.values[1:]:
            if t.shape != t_shape:
                msg = 'Component tensor {} has shape {}, others have {}'
                raise ValueError(msg.format(t.name, t.shape, t_shape))
        ret_shape = list(t_shape)
        ret_shape.insert(self.axis.val, num_tensors)
        return builtins.tensor(self.values[0].dtype, ret_shape)

    @precondition(allow=VALUE|SYMBOL)
    def value_inference(self):
        values = [v.val for v in self.values]
        return np.stack(values, self.axis.val)
