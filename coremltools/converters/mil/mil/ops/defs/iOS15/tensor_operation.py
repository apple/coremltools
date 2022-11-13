#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import math

import numpy as np

from coremltools.converters.mil.mil import (
    get_new_symbol,
    get_new_variadic_symbol,
    types,
)
from coremltools.converters.mil.mil.input_type import (
    DefaultInputs,
    InputSpec,
    ListOrTensorInputType,
    TensorInputType,
    TupleInputType,
)
from coremltools.converters.mil.mil.operation import (
    NONE,
    SYMBOL,
    VALUE,
    Operation,
    precondition,
)
from coremltools.converters.mil.mil.ops.defs._op_reqs import register_op
from coremltools.converters.mil.mil.ops.defs._utils import MAX_SIZE_CONSTANT_FOLDING
from coremltools.converters.mil.mil.types.symbolic import (
    any_symbolic,
    is_compatible_symbolic_vector,
    is_symbolic,
)


@register_op
class band_part(Operation):
    """
    Returns a tensor setting everything outside a center band to zeros for the innermost
    matrix. Special cases:

    - ``band_part(x, 0, -1)`` returns upper triangular part.
    - ``band_part(x, -1, 0)`` returns lower triangular part.
    - ``band_part(x, 0, 0)`` returns diagonal.

    Parameters
    ----------
    x: tensor<\*?, T> (Required)
        * Input tensor.
    lower: const<i32> (Optional)
        * Number of lower / below sub-diagonals to keep. If negative, keep entire
          lower triangle.
        * Defaults to ``-1`` (keep the entire lower triangle).
    upper: const<i32> (Optional)
        * Number of upper / above sub-diagonals to keep. If negative, keep entire
          lower triangle.
        * Defaults to ``-1`` (keep the entire upper triangle).

    Returns
    -------
    tensor<\*?, T>
        * Same type and shape as the input tensor.

    Attributes
    ----------
    T: fp16, fp32, i32, bool
    """

    input_spec = InputSpec(
        x=TensorInputType(type_domain="T"),
        lower=TensorInputType(const=True, optional=True, type_domain=types.int32),
        upper=TensorInputType(const=True, optional=True, type_domain=types.int32),
    )

    type_domains = {
        "T": (types.fp16, types.fp32, types.int32, types.bool),
    }

    def default_inputs(self):
        return DefaultInputs(
            lower=-1,
            upper=-1)

    def type_inference(self):
        return self.x.sym_type


@register_op
class cumsum(Operation):
    """
    Returns the cumulative sum of the input along the given axis.

    Parameters
    ----------
    x: tensor<\*?, T> (Required)
        * Input tensor.
    axis: const<i32> (Optional)
        * Defaults to ``0``.
        * Axis for which the cumulative sum is computed.
    exclusive: const<bool> (Optional)
        * Defaults to ``False``.
        * When set to ``False``, inclusive cumsum is computed, that is the first element of
          the output is identical to the first element in the input.
        * When set to ``True``, exclusive cumsum is computed, which makes the first element
          of output to ``0``.
    reverse: const<bool> (Optional)
        * Defaults to ``False``.
        * When set to ``True``, perform cumsum in the reverse order.

    Returns
    -------
    tensor<\*?, T>
        * Same type and shape as the input tensor.

    Attributes
    ----------
    T: fp16, fp32, i32
    """

    input_spec = InputSpec(
        x=TensorInputType(type_domain="T"),
        axis=TensorInputType(const=True, optional=True, type_domain=types.int32),
        exclusive=TensorInputType(const=True, optional=True, type_domain=types.bool),
        reverse=TensorInputType(const=True, optional=True, type_domain=types.bool),
    )

    type_domains = {
        "T": (types.fp16, types.fp32, types.int32),
    }

    def default_inputs(self):
        return DefaultInputs(
            axis=0,
            exclusive=False,
            reverse=False)

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
        # Check range of axis
        if self.axis.val < -1 or self.axis.val > self.x.rank - 1:
            raise ValueError(
                "axis should be in the range [-1, {}]".format(self.x.rank - 1)
            )

        return self.x.sym_type


@register_op
class fill(Operation):
    """
    Returns a tensor with a given shape filled with a constant value.

    Parameters
    ----------
    shape: tensor<[K], i32> (Required)
        * Target output tensor shape.
        * ``K`` is the rank of the output tensor. ``shape[k] > 0`` for ``k = 0,..., K-1``.
    value: const<T> (Optional)
        * Defaults to ``0.0``.
        * Constant value to fill in.

    Returns
    -------
    tensor<\*?, T>
        * Tensor with shape determined by the input shape.

    Attributes
    ----------
    T: fp16, fp32, i32, bool
    """

    input_spec = InputSpec(
        shape=TensorInputType(type_domain=types.int32),
        value=TensorInputType(const=True, optional=True, type_domain="T"),
    )

    type_domains = {
        "T": (types.fp16, types.fp32, types.int32, types.bool),
    }

    def default_inputs(self):
        return DefaultInputs(
            value=0.)

    def type_inference(self):
        if any_symbolic(self.shape.shape):
            # We can't infer any shape if shape has variable length.
            return types.tensor(self.value.dtype, (get_new_variadic_symbol(),))

        # shape has fixed length here.
        if self.shape.sym_val is None:
            ret_shape = tuple([get_new_symbol() for _ in range(self.shape.shape[0])])
            return types.tensor(self.value.dtype, ret_shape)

        return types.tensor(self.value.dtype, tuple(self.shape.sym_val.tolist()))

    @precondition(allow=VALUE)
    def value_inference(self):
        return np.full(shape=self.shape.val, fill_value=self.value.val)


@register_op
class non_maximum_suppression(Operation):
    """
    Applies non-maximum suppression (NMS) on the input box coordinates according
    to their intersection-over-union (IoU).

    NMS selects a subset of bounding boxes in descending order of score, and removes
    boxes that have high intersection-over-union (IOU) overlap with previously-selected
    boxes.


    Parameters
    ----------

    boxes: tensor<[n, B, 4], T> (Required)
        * Box coordinates on which to perform NMS.
    scores: tensor<[n, B, K], T> (Required)
        * Scores for each one of the boxes.
    iou_threshold: const<T> (Required)
        * The intersection over union (``IoU``) threshold over which boxes are
          suppressed. NMS remove all overlapping boxes with ``IoU > iou_threshold``.
    score_threshold: const<T> (Required)
        * Before IoU suppression is performed, boxes with class scores below this
          threshold are rejected.
    max_boxes: const<i32> (Required)
        * Maximum number of boxes to select. If the number of surviving boxes are
          less, output is padded up to this number.
    per_class_suppression: const<bool> (Optional)
        * Defaults to ``False``.
        * If ``True``, suppression is performed independently within boxes of each class.

    Returns
    -------
    tensor<[n, max_boxes, 4], T>
        * Coordinates of selected boxes.
    tensor<[n, max_boxes, K], T>
        * Scores of selected boxes.
    tensor<[n, max_boxes], i32>
        * Indices of selected boxes.
    tensor<[n], i32>
        * Number of boxes selected for each batch.

    Attributes
    ----------
    T: fp16, fp32
    """

    input_spec = InputSpec(
        boxes=TensorInputType(type_domain="T"),
        scores=TensorInputType(type_domain="T"),
        iou_threshold=TensorInputType(const=True, type_domain="T"),
        score_threshold=TensorInputType(const=True, type_domain="T"),
        max_boxes=TensorInputType(const=True, type_domain=types.int32),
        per_class_suppression=TensorInputType(const=True, optional=True, type_domain=types.bool),
    )

    type_domains = {
        "T": (types.fp16, types.fp32),
    }

    def default_inputs(self):
        return DefaultInputs(
            per_class_suppression=False)

    def type_inference(self):
        boxes_dtype = self.boxes.dtype
        scores_dtype = self.scores.dtype
        n_batch, _, n_score = self.scores.shape
        max_boxes = self.max_boxes.val

        return (
            types.tensor(boxes_dtype, (n_batch, max_boxes, 4)),
            types.tensor(scores_dtype, (n_batch, max_boxes, n_score)),
            types.tensor(types.int32, (n_batch, max_boxes)),
            types.tensor(types.int32, (n_batch,)),
        )


@register_op
class non_zero(Operation):
    """
    Returns the indices of the elements in the given tensor that are non-zero.

    Parameters
    ----------
    x: tensor<\*?, T> (Required)
        * Tensor, values selected at indices where its values is not equal to ``0``.

    Returns
    -------
    tensor<[N, R], int32>
        * 2-dimensional tensor contains indices of elements that are non-zero.
          Each row is the index for a non-zero value.
        * ``N`` is the number of non-zero elements, ``R`` is the rank of the input.

    Attributes
    ----------
    T: fp16, fp32, i32, bool
    """

    input_spec = InputSpec(
        x=TensorInputType(type_domain="T")
    )

    type_domains = {
        "T": (types.fp16, types.fp32, types.int32, types.bool),
    }

    def type_inference(self):
        shape = tuple([get_new_symbol(), self.x.rank])
        return types.tensor(types.int32, shape)

    @precondition(allow=VALUE)
    def value_inference(self):
        return np.transpose(np.nonzero(self.x.val))


@register_op
class one_hot(Operation):
    """
    Returns one-hot vectors whose locations represented in ``indices`` take the ``on_value``,
    while other locations take the ``off_value``.

    Parameters
    ----------
    indices: tensor<[D], i32> (Required)
        * Tensor, values indicate the locations for each one-hot vector to take the ``on_value``.
    one_got_vector_size: i32 (Required)
        * Indicates the number of returning vectors.
    axis: const i32 (Optional)
        * Indicates which dimension to append the new axis.
        * If the input indices is rank ``D``, the output tensor will have rank ``D+1``.
        * Defaults to ``-1`` (the last dimension).
    on_value: const T (Optional)
        * Values for locations where defined in ``indices``.
        * Defaults to ``1``.
    off_value: const T (Optional)
        * Defaults to ``0``.

    Returns
    -------
    tensor<\*?,T>
        * A tensor that contains one-hot vectors.

    Attributes
    ----------
    T: fp16, fp32, i32, bool
    """

    input_spec = InputSpec(
        indices=TensorInputType(type_domain=types.int32),
        one_hot_vector_size=TensorInputType(type_domain=types.int32),
        axis=TensorInputType(const=True, optional=True, type_domain=types.int32),
        on_value=TensorInputType(const=True, optional=True, type_domain="T"),
        off_value=TensorInputType(const=True, optional=True, type_domain="T"),
    )

    type_domains = {
        "T": (types.fp16, types.fp32, types.int32, types.bool),
    }

    def default_inputs(self):
        return DefaultInputs(
            axis=-1,
            on_value=1,
            off_value=0,
        )

    def type_inference(self):
        on_type = self.on_value.dtype
        off_type = self.off_value.dtype

        if on_type != off_type:
            raise TypeError(
                "Parameters on_value and off_value must have same input types."
            )

        if self.axis.val < -self.indices.rank - 1 or self.axis.val > self.indices.rank:
            raise IndexError(
                "Axis value {} is out of bounds for {} node {}".format(
                    self.axis.val, self.op_type, self.name
                )
            )

        indices_shape = list(self.indices.shape)

        depth_value = self.one_hot_vector_size.val
        if depth_value is None:
            depth_value = get_new_symbol()
        elif depth_value < 0:
            raise ValueError("Parameter one_hot_vector_size must be non-negative")

        retshape = indices_shape

        if self.axis.val < 0:
            cut = len(retshape) + self.axis.val + 1
        else:
            cut = self.axis.val
        retshape = retshape[0:cut] + [depth_value] + retshape[cut:]

        return types.tensor(on_type, retshape)


@register_op
class pad(Operation):
    """
    Pad a tensor.

    Parameters
    ----------

    x: tensor<[\*D_in], T>  (Required)

    pad: tensor<[2\*N], i32> (Required)
        ``N <= D_in``. Last ``N`` dimensions of ``x`` are padded as follows:

        * For each dimension ``i`` of ``x`` if ``i >= D_in - N``:
            * pad ``pad[2*i]`` elements before ``x[..,i,..]``.
            * pad ``pad[2*i+1]`` elements after ``x[..,i,..]``.
        * If mode is "reflect" then ``pad[2*i]`` and ``pad[2*i+1]`` can be at
          most ``D[i]-1``.
        * If mode is "replicate" then ``pad[2*i]`` and ``pad[2*i+1]`` can be
          at most ``D[i]``.

    mode: const<str> (Optional)
        * Defaults to ``constant``.
        * Must be one of the following values:
          ``constant``, ``reflect``, or ``replicate``.

    constant_val: const<T> (Optional)
        * Defaults to ``0``.
        * Constant value to pad. Ignored if ``mode != constant``.

    Returns
    -------
    tensor<[\*D_out],T>
        * Tensor with same type as the input.

    Attributes
    ----------
    T: fp16, fp32
    """

    input_spec = InputSpec(
        x=TensorInputType(type_domain="T"),
        pad=TensorInputType(type_domain=types.int32),
        mode=TensorInputType(const=True, optional=True, type_domain=types.str),
        constant_val=TensorInputType(const=True, optional=True, type_domain="T"),
    )

    type_domains = {
        "T": (types.fp16, types.fp32),
    }

    def default_inputs(self):
        return DefaultInputs(
            mode="constant",
            constant_val=0.,
        )

    def type_inference(self):
        in_shape = self.x.shape
        ret_shape = list(in_shape)
        pad = self.pad
        if len(pad.shape) != 1:
            raise ValueError("Pad should be a 1D tensor!")
        if self.mode and not self.mode.val in {'constant', 'reflect', 'replicate'}:
            raise ValueError("Pad mode should be one of {'constant', 'reflect', 'replicate'}")

        if pad.val is None:
            for i in range(self.pad.shape[0] // 2):
                ret_shape[-self.pad.shape[0] // 2 + i] = get_new_symbol()
        else:
            pad = pad.val
            pad = pad.copy()

            if len(pad) % 2 != 0:
                raise ValueError("Number of elements in the argument Pad must be divisible by 2.")

            pad = pad.reshape(-1, 2)

            if pad.shape[0] > len(ret_shape):
                raise ValueError(
                    "Number of dimensions specified through pad must less than or equal to rank "
                    "of input x"
                )

            for i in range(len(pad)):
                ret_shape[-len(pad) + i] = ret_shape[-len(pad) + i] + pad[i][0] + pad[i][1]

        return types.tensor(self.x.dtype, tuple(ret_shape))

    @precondition(allow=VALUE)
    def value_inference(self):
        # NumPy `edge` mode is equivalent to `replicate` mode of PyTorch and CoreML
        mode = "edge" if self.mode.val == "replicate" else self.mode.val
        pad_val = self.pad.val

        if pad_val is None:
            return None

        if len(self.x.val.shape) > (pad_val.shape[0] // 2):
            updated_pad = np.zeros(len(self.x.val.shape) * 2)
            updated_pad[-pad_val.shape[0] :] = pad_val
            pad_val = updated_pad
        pad_val = pad_val.reshape(-1, 2).astype(np.int32)
        if mode == "constant":
            return np.pad(
                self.x.val, pad_val, mode, constant_values=self.constant_val.val
            )
        # NumPy does not support non-constant mode and constant_values argument
        return np.pad(self.x.val, pad_val, mode)


@register_op
class range_1d(Operation):
    """
    Returns a numpy-like 1-D range sequence.

    Parameters
    ----------
    end: <T> (Required)
        * The upper limit of the sequence, exclusive.
    start: <T> (Required)
        * The start point of the sequence.
    step: <T> (Required)
        * Number that increments ``start``.

    Returns
    -------
    tensor<M, T>
        * A 1-D tensor, where ``M`` is the length of the sequence.

    Attributes
    ----------
    T: i32, fp16, fp32
    """

    input_spec = InputSpec(
        end=TensorInputType(type_domain="T"),
        start=TensorInputType(type_domain="T"),
        step=TensorInputType(type_domain="T"),
    )

    type_domains = {
        "T": (types.fp16, types.fp32, types.int32),
    }

    @precondition(allow=VALUE)
    def value_inference(self):
        start = self.start.val
        end = self.end.val
        step = self.step.val
        shape = (end - start) / step
        # To prevent from creating constant greater then 1MB,
        # a upper bound of the size of the resulting array is set.
        if shape > MAX_SIZE_CONSTANT_FOLDING:
            return None
        return np.arange(start, end, step)

    def type_inference(self):
        start = self.start.sym_val
        end = self.end.sym_val
        step = self.step.sym_val

        if (
            (self.start.dtype != self.end.dtype)
            or (self.start.dtype != self.step.dtype)
            or (self.end.dtype != self.step.dtype)
        ):
            raise TypeError(
                "All inputs to the range operation must have same input types."
            )

        if all(sym_val is not None for sym_val in (start, end, step)):
            shape = (end - start) / step
            shape = shape if is_symbolic(shape) else int(math.ceil(shape))
            shape = tuple([shape])
        else:
            shape = tuple(
                [
                    get_new_symbol(),
                ]
            )

        return types.tensor(self.start.dtype, shape)


@register_op
class tile(Operation):
    """
    Returns a new tensor by replicating input ``x`` multiples times.
    Dimension ``i`` of ``x`` will be replicated ``reps[i]`` times.

    Parameters
    ----------
    x: tensor<\*?, T> (Required)
        * Input tensor.
    reps: tensor<[rank(x)], i32> (Required)
        * A 1-D tensor with length ``rank(x)``, which indicates the number to replicate the input along each dimension.

    Returns
    -------
    tensor<\*?, T>:
        * An n-D tensor with same type as the input.

    Attributes
    ----------
    T: fp16, fp32, i32, bool
    """

    input_spec = InputSpec(
        x=TensorInputType(type_domain="T"),
        reps=TensorInputType(type_domain=types.int32),
    )

    type_domains = {
        "T": (types.fp16, types.fp32, types.int32, types.bool),
    }

    def type_inference(self):
        x_type = self.x.dtype
        x_shape = np.array(self.x.shape)

        reps = self.reps.sym_val

        if reps is None:
            out_shape = tuple([get_new_symbol() for _ in range(self.x.rank)])
            return types.tensor(x_type, out_shape)

        if len(reps) == 0 or len(reps) != self.x.rank:
            msg = (
                "Length of the reps ({}) must be at least 1, and "
                "equal to the rank of the input x ({})"
            )
            raise ValueError(msg.format(len(reps), self.x.rank))

        out_shape = []
        for i, rep in enumerate(reps):
            if not is_symbolic(rep):
                if rep <= 0:
                    raise ValueError("All entries of reps parameter must be greater than 0")

            if is_symbolic(rep) or is_symbolic(x_shape[i]):
                out_shape.append(get_new_symbol())
            else:
                out_shape.append(rep * x_shape[i])

        out_shape = tuple(out_shape)

        return types.tensor(x_type, out_shape)

    @precondition(allow=VALUE)
    def value_inference(self):
        # Infer only if don't have symbolic values.
        if self.reps.val is None:
            return None
        return np.tile(self.x.val, reps=self.reps.val)


@register_op
class argsort(Operation):
    """
    Returns a tensor containing the indices of the sorted values along a given axis
    of the input tensor.

    Parameters
    ----------
    x: <\*?, T> (Required)
        * Input tensor.
    * axis: const<i32> (Optional)
        * Defaults to ``-1`` (the last dimension).
        * Axis to perform the operation.
    * ascending: const<bool> (Optional)
        * Defaults to ``False``, sort in descending order.
        * ``True`` to sort in ascending order.

    Returns
    -------
    tensor<\*?, int32>
        * Tensor containing the indices of the sorted values

    Attributes
    ----------
    T: fp16, fp32, i32
    """

    input_spec = InputSpec(
        x=TensorInputType(type_domain="T"),
        axis=TensorInputType(const=True, optional=True, type_domain=types.int32),
        ascending=TensorInputType(const=True, optional=True, type_domain=types.bool),
    )

    type_domains = {
        "T": (types.fp16, types.fp32, types.int32),
    }

    def default_inputs(self):
        return DefaultInputs(
            axis=-1,
            ascending=False,
        )

    def type_inference(self):
        return types.tensor(types.int32, self.x.shape)

    @precondition(allow=VALUE)
    def value_inference(self):
        # The default np argsort mode is ascending, which is opposite to MIL's argsort op.
        if self.ascending.val:
            return np.argsort(self.x.val, axis=self.axis.val)
        return np.argsort(-self.x.val, axis=self.axis.val)


@register_op
class topk(Operation):
    """
    Returns a tensor containing top or bottom ``k`` values and the corresponding
    indices of the input tensor along a given axis.

    Parameters
    ----------
    x: <\*?, T> (Required)
        * Input tensor.
    k: const<i32> (Optional)
        * Defaults to ``1``.
        * Number of values/indices to be computed along each axis.
    axis: const<i32> (Optional)
        * Defaults to ``-1`` (last dimension).
        * Axis to perform the operation.
    ascending: const<bool> (Optional)
        * Defaults to ``False``, sort in descending order.
        * ``True`` to sort in ascending order.

    Returns
    -------
    tensor<\*?, T>
        * Values of top/bottom ``k`` elements.
    tensor<\*?, int32>
        * Indices of the top/bottom ``k`` elements along axis.

    Attributes
    ----------
    T: fp16, fp32, int32
    """

    input_spec = InputSpec(
        x=TensorInputType(type_domain="T"),
        k=TensorInputType(const=True, optional=True, type_domain=types.int32),
        axis=TensorInputType(const=True, optional=True, type_domain=types.int32),
        ascending=TensorInputType(const=True, optional=True, type_domain=types.bool),
    )

    type_domains = {
        "T": (types.fp16, types.fp32, types.int32),
    }

    def default_inputs(self):
        return DefaultInputs(
            k=1,
            axis=-1,
            ascending=False,
        )

    def type_inference(self):
        x_type = self.x.dtype
        x_shape = self.x.shape
        k = self.k.val
        axis = self.axis.val

        if not is_symbolic(x_shape[axis]) and k > x_shape[axis]:
            msg = "K={} is greater than size of the given axis={}"
            raise ValueError(msg.format(k, axis))

        ret_shape = list(x_shape)
        ret_shape[axis] = k
        return types.tensor(x_type, ret_shape), types.tensor(types.int32, ret_shape)

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


@register_op
class flatten2d(Operation):
    """
    Flattens input tensor into 2d tensor by flattening dimensions before and
    after the provided axis.

    Parameters
    ----------
    x: tensor<[*d], T> (Required)
        * Input tensor.
    axis: const<i32>  (Optional)
        * Defaults to ``1``.
        * Negative axis is supported.

    Returns
    -------
    tensor<d_prior, d_post, T>
        * ``d_prior`` is product of dimensions ``x[:axis]``
        * ``d_post`` is product of dimensions ``x[axis:]``

    Examples
    --------
        1. ``input_shape = (3, ), axis = -1, output_shape = (1, 3)``
        2. ``input_shape = (3, ), axis = 1, output_shape = (3, 1)``
        3. ``input_shape = (4, 3), axis = -1, output_shape = (4, 3)``
        4. ``input_shape = (2, 3, 2), axis = -1, output_shape = (6, 2)``
        5. ``input_shape = (5, 5, 2), axis = 1, output_shape = (5, 10)``

    Attributes
    ----------
    T: fp16, fp32, i32, bool
    """

    input_spec = InputSpec(
        x=TensorInputType(type_domain="T"),
        axis=TensorInputType(const=True, optional=True, type_domain=types.int32)
    )

    type_domains = {
        "T": (types.fp16, types.fp32, types.int32, types.bool),
    }

    def default_inputs(self):
        return DefaultInputs(
            axis=1,
        )

    def type_inference(self):
        shape = list(self.x.shape)
        axis = self.axis.val
        dim_pre_axis = np.prod(shape[:axis])
        dim_post_axis = np.prod(shape[axis:])
        new_shape = [dim_pre_axis, dim_post_axis]
        return types.tensor(self.x.dtype, tuple(new_shape))

    @precondition(allow=VALUE | SYMBOL)
    def value_inference(self):
        shape = self.x.shape
        axis = self.axis.val

        dim_pre_axis = np.prod(shape[:axis])
        dim_post_axis = np.prod(shape[axis:])
        return self.x.val.reshape(dim_pre_axis, dim_post_axis)


@register_op
class shape(Operation):
    """
    Returns a 1-dimensional tensor with the shape of the input tensor.

    Parameters
    ----------
    x: tensor<[*?], T> (Required)
        * Input tensor.

    Returns
    -------
    tensor<K, i32>
        * Shape of the input tensor.
        * ``K = x.rank``.

    Attributes
    ----------
    T: fp16, fp32, i32, bool
    """

    input_spec = InputSpec(x=TensorInputType(type_domain="T"))

    type_domains = {
        "T": (types.fp16, types.fp32, types.int32, types.bool),
    }

    def type_inference(self):
        input_rank = self.x.rank
        return types.tensor(types.int32, tuple([input_rank]))

    def value_inference(self):
        if any_symbolic(self.x.shape):
            # convert elements in shape to int32
            res = [x if is_symbolic(x) else np.int32(x) for x in self.x.shape]
            return np.array(res)
        else:
            return np.array(self.x.shape).astype(np.int32)


@register_op
class concat(Operation):
    """
    Concatenates tensors along a dimension.

    Parameters
    ----------
    values: Tuple[tensor<[d0, d1, ..., d_axis_i, ..., d_n],T>] (Required)
        * The number of dimensions of the input tensors must match, and all
          dimensions except ``axis`` must be equal.
        * The tensors may be variadic, but the number of tensors must be
          determined at compile time (i.e. a tuple).
    axis: const<int32> (Required)
        * The dimension along which to concatenate. Must be in the range
          ``[-rank(values[i]), rank(values[i]))`` for all ``i``.
    interleave: const<bool> (Optional, Default=False)
        * If True, concatenate the inputs by interleaving them.
        * If True, all the inputs to this op must have the exact same shape.

    Examples
    --------

    .. sourcecode:: python

        in1 = [[1, 2], [3, 4], [5, 6]]  # shape (3, 2)
        in2 = [[7, 8], [9, 10], [11, 12]]  # shape (3, 2)
        axis = 0  # output shape is (6, 2)

        if interleave is False:  # default
            # output[0:3, :] = in1
            # output[3:6, :] = in2
            output = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]]

        if interleave is True:
            # output[0::2, :] = in1
            # output[1::2, :] = in2
            output = [[1, 2], [7, 8], [3, 4], [9, 10], [5, 6], [11, 12]]

    Returns
    -------
    tensor<[d0, d1,...d_axis_out, ..., d_n],T>
        * Where ``d_axis_out = sum(d_axis_i)``.

    Attributes
    ----------
    T: fp16, fp32, i32, bool
    """

    input_spec = InputSpec(
        values=TupleInputType(),
        axis=TensorInputType(const=True, type_domain=types.int32),
        interleave=TensorInputType(const=True, optional=True, type_domain=types.bool)
    )

    def default_inputs(self):
        return DefaultInputs(
            interleave=False,
        )

    def type_inference(self):
        concat_dim_len = 0
        if len(self.values) == 0:
            raise ValueError("Concat {} got 0 values".format(self.name))

        # Validate values have the same rank
        rank = self.values[0].rank
        for v in self.values:
            if v.rank != rank:
                msg = "Input {} has rank {} != other inputs rank {}"
                raise ValueError(msg.format(v.name, v.rank, rank))

        # Check concat axis is within (-rank, rank)
        concat_axis = self.axis.val
        if concat_axis < 0:
            concat_axis += rank
        if rank > 0 and (concat_axis < 0 or concat_axis >= rank):
            msg = "In {} of op_type {}: axis out of bound for input " + "(rank {})"
            raise ValueError(msg.format(self.name, self.op_type, rank))

        # Validate values share the same data type
        dtype = self.values[0].dtype
        for v in self.values[1:]:
            if v.dtype != dtype:
                msg = (
                    "Tensors in 'values' of the concat op ({}) should share the "
                    "same data type. Got {}."
                ).format(self.name, [x.dtype for x in self.values])
                raise ValueError(msg)

        # validate that non-axis dimensions match
        retshape = list(self.values[0].shape)
        for v in self.values[1:]:
            for i in range(rank):
                if is_symbolic(retshape[i]) or is_symbolic(v.shape[i]):
                    continue
                if i != concat_axis and retshape[i] != v.shape[i]:
                    msg = 'Dimension mismatch in {} ("{}"): shapes {} vs. {}'
                    raise ValueError(
                        msg.format(self.op_type, self.name, retshape, v.shape)
                    )
                if self.interleave.val and retshape[i] != v.shape[i]:
                    msg = 'Dimension mismatch in {} ("{}"): shapes {} vs. {}. ' \
                          'All inputs must have same shape when \'interleave\' option is True.'
                    raise ValueError(
                        msg.format(self.op_type, self.name, retshape, v.shape)
                    )

        # Get length of concat dim
        concat_dim_len = 0
        for v in self.values:
            if len(v.shape) == 0:
                taxis = 1
            else:
                taxis = v.shape[concat_axis]
            if is_symbolic(taxis):
                concat_dim_len = get_new_symbol()
                break
            concat_dim_len += taxis

        if len(retshape) == 0:
            retshape = [concat_dim_len]
        else:
            retshape[concat_axis] = concat_dim_len

        return types.tensor(dtype, retshape)

    @precondition(allow=VALUE | SYMBOL | NONE)
    def value_inference(self):

        values = []
        for v in self.values:
            if v.sym_val is not None:
                values.append(v.sym_val)
                continue
            if v.rank == 0:
                values.append(get_new_symbol())
                continue
            if any_symbolic(v.shape):
                values.append(None)
                continue

            # we support value inference when number of elements for each tensor is less than 10
            shape = v.shape
            num_element = np.prod(shape)
            if num_element > 10:
                values.append(None)
                continue

            symbolic_tensor = [get_new_symbol() for _ in range(num_element)]
            symbolic_tensor = np.reshape(np.array(symbolic_tensor), shape)
            values.append(symbolic_tensor)

        if any([val is None for val in values]):
            return None

        if not isinstance(values[0], np.ndarray) or values[0].shape == ():
            return np.stack(values, axis=self.axis.val)

        return np.concatenate(values, axis=self.axis.val)


@register_op
class split(Operation):
    """
    Split tensors into a tuple

    Parameters
    ----------
    x: <\*?,T>  (Required)
        * The tensor to split.
        * The tensors may be variadic, but the number of tensors must be determined
          at compile time (i.e. a tuple).

    num_splits: <i32> (Optional)
        If specified, divide ``x`` into ``num_splits`` tensors along ``axis``.
        Its behavior depends on ``split_sizes``:

            * If ``split_sizes`` is defined, ``num_splits == S``, and the output
              sizes may be uneven.
            * If ``split_sizes`` is not defined, ``value.shape[axis]`` must be
              divisible by ``num_splits``, and the output sizes must be even.

        At least one of ``num_splits`` or ``split_sizes`` must be provided.
        If ``split_sizes`` length ``S`` cannot be determined at compile time,
        ``num_splits`` must be supplied to determine the number of outputs.

    split_sizes: const<S, i32> (Optional)
        * Sizes to split to. The sum of ``split_sizes`` must equal to
          ``value.shape[axis]``.

    axis: const<i32> (Required)
        * The dimension along which to concatenate. Must be in the
          range ``[-rank(x), rank(x))``.

    Returns
    -------
    Tuple[tensor<\*?, T>]
        * Where the length of the tuple is the number of splits (determined
          from ``num_splits`` or ``split_sizes``).

    Attributes
    ----------
    T: fp16, fp32, i32, bool
    """

    input_spec = InputSpec(
        x=TensorInputType(type_domain="T"),
        num_splits=TensorInputType(const=True, optional=True, type_domain=types.int32),
        split_sizes=TensorInputType(const=True, optional=True, type_domain=types.int32),
        axis=TensorInputType(const=True, type_domain=types.int32),
    )

    type_domains = {
        "T": (types.fp16, types.fp32, types.int32, types.bool),
    }

    def type_inference(self):
        num_splits, sizes = self._get_num_splits_and_sizes()
        x_shape = list(self.x.shape)
        ret_shapes = [x_shape[:] for _ in range(num_splits)]
        axis = self.axis.val
        for i, d in enumerate(sizes):
            ret_shapes[i][axis] = d
        self.sizes = sizes
        return tuple([types.tensor(self.x.dtype, s) for s in ret_shapes])

    def _get_num_splits_and_sizes(self):
        """
        Return:
        - num_splits: int
        - sizes: list of int/symbols. Of length num_splits

        Raise ValueError if num_splits cannot be determined.
        """
        if self.num_splits is None and self.split_sizes is None:
            msg = (
                "At least one of num_splits and split_sizes "
                + "must be specified in split op {}"
            )
            raise ValueError(msg.format(self.name))

        axis = self.axis.val

        if self.num_splits is not None:
            num_splits = self.num_splits.val
            if self.split_sizes is None:
                # Even split
                if (
                    not is_symbolic(self.x.shape[axis])
                    and self.x.shape[axis] % num_splits != 0
                ):
                    msg = "num_split {} does not divide split " + "dim (length = {})"
                    raise ValueError(msg.format(num_splits, self.x.shape[axis]))
                size = self.x.shape[axis] / num_splits
                return num_splits, [size] * num_splits

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
            raise ValueError("Unable to determine number of splits")

        num_splits = len(self.split_sizes.shape)
        sizes = [get_new_symbol() for _ in range(num_splits)]
        return num_splits, sizes

    @precondition(allow=VALUE | SYMBOL | NONE)
    def value_inference(self):
        num_splits, sizes = self._get_num_splits_and_sizes()
        if self.x.sym_val is None or any_symbolic(sizes):
            raise NotImplementedError()

        if num_splits == 1:
            # No split_indices possible.
            return self.x.sym_val

        split_indices = np.cumsum(sizes).astype(np.int)
        return tuple(np.split(self.x.sym_val, split_indices[:-1], axis=self.axis.val))


@register_op
class stack(Operation):
    """
    Concatenates tensors along a dimension.

    Parameters
    ----------
    values: Tuple[tensor<[d0, d1,...d_axis_i, ..., d_n], T>]  (Required)
        * All tensors must have identical shape.
    axis: const<i32> (Required)
        * The dimension along which to concatenate. Must be in the range ``[-rank(values[i]), rank(values[i]))`` for all ``i``.

    Returns
    -------
    tenor<[d0, d1,...d_axis_out, ..., d_n], T>
        * Where ``d_axis_out = sum(d_axis_i)``.

    Attributes
    ----------
    T: fp16, fp32, i32, bool
    """

    input_spec = InputSpec(
        values=TupleInputType(),
        axis=TensorInputType(const=True, type_domain=types.int32)
    )

    def type_inference(self):

        num_tensors = len(self.values)
        if num_tensors == 0:
            raise ValueError("Cannot stack 0 tensor")

        # get the first value without symbolic shape
        t_shape = None
        for value in self.values:
            if not any_symbolic(value.shape):
                t_shape = value.shape
                break
        t_shape = self.values[0].shape if t_shape is None else t_shape

        # compare all shape
        for t in self.values:
            if not is_compatible_symbolic_vector(t.shape, t_shape):
                msg = "Component tensor {} has shape {}, others have {}"
                raise ValueError(msg.format(t.name, t.shape, t_shape))

        # Validate values share the same data type
        dtype = self.values[0].dtype
        for v in self.values[1:]:
            if v.dtype != dtype:
                msg = (
                    "Tensors in 'values' of the stack op ({}) should share the "
                    "same data type. Got {}."
                ).format(self.name, [x.dtype for x in self.values])
                raise ValueError(msg)

        axis = self.axis.val
        if axis < 0:
            axis += (self.values[0].rank + 1)
        ret_shape = list(t_shape)
        ret_shape.insert(axis, num_tensors)
        return types.tensor(self.values[0].dtype, ret_shape)

    @precondition(allow=VALUE | SYMBOL | NONE)
    def value_inference(self):

        is_all_rank_zero = all([v.rank == 0 for v in self.values])
        values = [
            v.sym_val if v.sym_val is not None else get_new_symbol()
            for v in self.values
        ]

        if any([is_symbolic(v) for v in values]) and not is_all_rank_zero:
            return None

        return np.stack(values, self.axis.val)


# identity is used for renaming and is rarely necessary. See
# `loop_invariant_elimination` pass for a rare use case.
@register_op
class identity(Operation):
    """
    Returns a tensor with the same shape and contents as input.

    Parameters
    ----------
    x: tensor<\*?, T> (Required)
        * Input tensor.

    Returns
    -------
    tensor<\*?, T>
        * Same type and shape as the input tensor.

    Attributes
    ----------
    T: fp16, fp32, i32, bool
    """

    input_spec = InputSpec(
        x=ListOrTensorInputType()
    )

    def type_inference(self):
        return self.x.sym_type

    @precondition(allow=VALUE | SYMBOL)
    def value_inference(self):
        return self.x.sym_val
