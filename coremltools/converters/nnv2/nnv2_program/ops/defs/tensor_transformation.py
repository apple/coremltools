import functools
import sympy as sm
from coremltools.converters.nnv2.builtin_types.symbolic import (
        is_symbolic, isscalar, any_symbolic, any_variadic)
from coremltools.converters.nnv2.nnv2_program.program.program import get_new_symbol, get_new_variadic_symbol, SYMBOL
from ._op_reqs import *

@register_op(doc_str="""
Rearranges elements in a tensor from depth (channel) into spatial dimensions.

Inputs

* x: <n, C, H, W, T> Required
    * Input tensor of rank 4.
* block_size: const<i32> Required
    * The size of the spatial block. Must be greater than 1 and divisible by channel dimension.

Outputs

* <n, C / block_size^2, H x block_size, W x block_size, T> where b is the block size.

Type Domains

* T: f32
""")
class depth_to_space(Operation):
    input_spec = InputSpec(
        x=TensorInputType(),
        block_size=IntInputType(const=True),
    )

    def __init__(self, **kwargs):
        super(depth_to_space, self).__init__(**kwargs)

    def type_inference(self):
        x_type = self.x.dtype
        n, c, h, w = self.x.shape
        bs = self.block_size.val
        ret_shape = (n, c // (bs * bs), h * bs, w * bs)
        return builtins.tensor(x_type, ret_shape)


# We use markdown syntax. Italicizes (Required), (Optional...). Use e.g.,
# $\math{R}$ for math expression.
@register_op(doc_str="""
# expand_dims

Insert a single-dimension in a 1D or higher tensor at index axis.

### Inputs

- `x: <*,T>` _(Required)_ Scalar or tensor
- `axis: const<i32>` _(Required)_ Insert single dimension at axis dimension
index. Negative value to index from the end. `-D-1 <= axis <= D` where `D` is
the rank of `x`.

### Outputs

- `<*,T>` tensor of 1D or higher.

### Type Domains

- T: f32
""")
class expand_dims(Operation):
    input_spec = InputSpec(
            x = TensorInputType(),
            axis = IntInputType(const=True),
            )

    def __init__(self, **kwargs):
        super(expand_dims, self).__init__(**kwargs)

    def type_inference(self):
        x_type = self.x.dtype
        x_shape = list(self.x.shape)
        if self.axis.val < -self.x.rank - 1 \
                or self.axis.val > self.x.rank:
            raise IndexError(
                'Axis value {} is out of bounds for {} node {}'.format(
                    self.axis.val, self.op_type, self.name))
        cut = self.axis.val
        if cut < 0:
            cut = self.x.rank + cut + 1
        ret_shape = x_shape[:cut] + [1] + x_shape[cut:]

        return builtins.tensor(x_type, tuple(ret_shape))

    @precondition(allow=VALUE)
    def value_inference(self):
        return np.expand_dims(self.x.val, axis=self.axis.val)


def reshape_with_symbol(v, shape):
    """
    Perform basic reshape if v is symbolic (not array of symbols).
    """
    if is_symbolic(v):
        return np.array(v).reshape(shape)
    shape = [int(s) for s in shape]
    return v.reshape(shape)


# rdar://58622145
@register_op(doc_str='TODO')
class reshape(Operation):
    input_spec = InputSpec(
            x = TensorInputType(),
            shape = IntTensorInputType(),
            )

    def __init__(self, **kwargs):
        super(reshape, self).__init__(**kwargs)

    def type_inference(self):
        if any_symbolic(self.shape.shape):
            # We can't infer any shape if shape has variable length.
            return builtins.tensor(self.x.dtype, (get_new_variadic_symbol(),))

        # shape has fixed length here.
        if self.shape.sym_val is None:
            shape = tuple([get_new_symbol() for _ in range(self.shape.shape[0])])
            return builtins.tensor(self.x.dtype, shape)
        t, _ = self._get_type_val()
        return t

    @precondition(allow=VALUE|SYMBOL)
    def value_inference(self):
        _, val = self._get_type_val()
        return val

    def _get_type_val(self):
        x_type = self.x.dtype
        x_shape = self.x.shape
        x_vol = functools.reduce(lambda a,b : a*b, x_shape)
        # shape is const, and thus sym_val is not None
        sym_shape = self.shape.sym_val
        sym_shape = [get_new_symbol() if d == -1 else d for d in sym_shape]
        ret_shape = reshape.enforce_volumetric_constraint(x_vol, sym_shape)
        ret_val = None
        if self.x.val is not None and \
            all(isscalar(a) for a in ret_shape):
            ret_val = reshape_with_symbol(self.x.val, ret_shape)
        return builtins.tensor(x_type, tuple(ret_shape)), ret_val

    @staticmethod
    def enforce_volumetric_constraint(left_volume, inshape):
        left_symbols = set()
        if is_symbolic(left_volume):
            left_symbols = left_volume.free_symbols
        # Generally, we want to solve for right in terms of left. But this
        # is kinda annoying actually.
        shape = list(inshape)

        # TODO: <rdar://problem/58583272> NNV2: implement missing functionality for Reshape op
        # Handling when reshape is given 0 instead of actual input
        # input tensor shape: [4, 3, 2], reshape:[0, -1], output tensor shape: [4, 6]
        if shape.count(-1) > 1:
            raise ValueError("Reshape op supports only one dimension to be -1. Given {}".format(shape.count(-1)))

        infer_dim_index = shape.index(-1) if -1 in shape else None
        right_volume = 1
        for i in shape:
            if i != -1:
                right_volume = right_volume * i

        if infer_dim_index:
            shape[infer_dim_index] = left_volume // right_volume

        if not is_symbolic(right_volume):
            return shape

        constraints = [left_volume - right_volume]
        solve_for = [s for s in shape if is_symbolic(s)]

        for rightsym in solve_for:
            sol = sm.solve(constraints, [rightsym], dict=True)
            if not isinstance(sol, list):
                sol = [sol]
            # look for an acceptable solution
            for s in sol:
                if 0 in s.values():
                    continue
                for i in range(len(shape)):
                    if shape[i] in s:
                        v = s[shape[i]]
                        if len(v.free_symbols - left_symbols) > 0:
                            continue
                        try:
                            shape[i] = int(v)
                        except:
                            shape[i] = v
        return shape


@register_op(doc_str="""
Reverses the order of the input tensor along specified axes / dimensions.

Inputs

* x: <*, T> Required
    * Input tensor.
* axes: const<D, i32> Optional
    * Dimension(s) to reverse. Each axis must be in the range [-rank(x), rank(x)).
    * Defaults to None (reduce on all dimensions).

Outputs

* <*, T> same type as the input tensor.

Type Domains

* T: f32
""")
class reverse(Operation):
    input_spec = InputSpec(
        x=TensorInputType(),
        axes=IntTensorInputType(const=True, optional=True),
    )

    def __init__(self, **kwargs):
        super(reverse, self).__init__(**kwargs)

    def type_inference(self):
        return self.x.sym_type

    @precondition(allow=VALUE)
    def value_inference(self):
        res = self.x.val
        axes = self.axes.val if self.axes is not None else range(self.x.rank)
        for axis in axes:
            res = np.flip(res, axis=axis)
        return res


@register_op(doc_str="""
Reverses variable length slices for specified axes / dimensions of the input
tensor. This op first slice input tensor along the batch_axis dimension, then
partially reverse the elements along the seq_axis for the first lengths[i]
elements.

Inputs

* x: <*, T> Required
    * Input tensor.
* lengths: const<L, i32> Required
    * 1-dimensional tensor of length x.shape[batch_axis] specifying the length
    of the sequence to reverse.
    * Values must be in range [0, x.shape[seq_axis]).
* seq_axis: const<i32> Optional
    * The dimension to reverse.
    * Defaults to 0.
* batch_axis: const<i32> Optional
    * Dimension for slicing.
    * Defaults to 0.

Outputs

* <*, T> same type as the input tensor.

Type Domains

* T: f32
""")
class reverse_sequence(Operation):
    input_spec = InputSpec(
        x=TensorInputType(),
        lengths=IntTensorInputType(),
        seq_axis=IntInputType(const=True, default=0),
        batch_axis=IntInputType(const=True, default=0)
    )

    def __init__(self, **kwargs):
        super(reverse_sequence, self).__init__(**kwargs)

    def type_inference(self):
        return self.x.sym_type

    @precondition(allow=VALUE)
    def value_inference(self):
        raise NotImplementedError('TODO')


@register_op(doc_str='TODO')
class slice_by_size(Operation):
    input_spec = InputSpec(
        x=TensorInputType(),
        begin=IntTensorInputType(),
        size=IntTensorInputType(const=True),
    )

    def __init__(self, **kwargs):
        super(slice_by_size, self).__init__(**kwargs)

    def type_inference(self):
        if self.begin.rank != 1:
            raise ValueError("begin should be 1-D tensor, got {}-D tensor instead".format(self.begin.rank))
        if self.size.rank != 1:
            raise ValueError("size should be 1-D tensor, got {}-D tensor instead".format(self.size.rank))
        if self.x.rank != self.begin.shape[0]:
            raise ValueError("Length of begin {} doesn't equal to input rank {}.".format(len(self.begin.shape[0]), len(self.x.rank)))
        if self.x.rank != self.size.shape[0]:
            raise ValueError("Length of size {} doesn't equal to input rank {}.".format(len(self.size.shape[0]), len(self.x.rank)))

        x_shape = self.x.shape
        ret_shape = []
        for idx, s in enumerate(self.size.val):
            if s != -1:
                ret_shape.append(s)
            elif self.begin.sym_val is not None:
                ret_shape.append(x_shape[idx]-self.begin.sym_val[idx])
            else:
                ret_shape.append(get_new_symbol())

        return builtins.tensor(self.x.dtype, tuple(ret_shape))


@register_op(doc_str="""
Rearranges elements in a tensor from spatial into depth (channel) dimension.

Inputs

* x: <n, C, H, W, T> Required
    * Input tensor of rank 4.
* block_size: const<i32> Required
    * The size of the spatial block. Must be greater than 1 and divisible by spatial dimensions H, W.

Outputs

* <n, C x block_size^2, H / block_size, W / block_size, T> where b is the block size.

Type Domains

* T: f32
""")
class space_to_depth(Operation):
    input_spec = InputSpec(
        x=TensorInputType(),
        block_size=IntInputType(const=True),
    )

    def __init__(self, **kwargs):
        super(space_to_depth, self).__init__(**kwargs)

    def type_inference(self):
        x_type = self.x.dtype
        n, c, h, w = self.x.shape
        bs = self.block_size.val
        ret_shape = (n, c * (bs * bs), h // bs, w // bs)
        return builtins.tensor(x_type, ret_shape)


# rdar://58622145
@register_op(doc_str='TODO')
class squeeze(Operation):
    input_spec = InputSpec(
            x = TensorInputType(),
            axes = IntTensorInputType(const=True, optional=True),
            )

    def __init__(self, **kwargs):
        super(squeeze, self).__init__(**kwargs)

    def type_inference(self):
        x_type = self.x.dtype
        x_shape = self.x.shape
        squeezed_shape = list(x_shape)
        if self.axes is None:
            # Squeeze all single-dim, assuming symbolic dims != 1
            squeezed_shape = [s for s in squeezed_shape if s != 1]
        else:
            axes = self.axes.val
            axes = [axis if axis >=0 else axis + self.x.rank for axis in axes]
            for i in sorted(axes)[::-1]: # descending order
                if len(squeezed_shape) <= i:
                    raise ValueError("Cannot squeeze dim {} for shape"+
                            " {}".format(i, squeezed_shape))
                squeezed_shape.pop(i)

        return builtins.tensor(x_type, tuple(squeezed_shape))

    @precondition(allow=VALUE|SYMBOL)
    def value_inference(self):
        return np.squeeze(self.x.val, axis=tuple(self.axes.val))


# rdar://58622145
@register_op(doc_str='TODO')
class transpose(Operation):
    input_spec = InputSpec(
            x = TensorInputType(),
            perm = IntTensorInputType(const=True),
            )

    def __init__(self, **kwargs):
        super(transpose, self).__init__(**kwargs)

    def type_inference(self):
        x_type = self.x.dtype
        perm = self.perm.val
        x_shape = np.array(self.x.shape)
        if len(perm) != len(x_shape):
            raise ValueError("perm should have the same length as rank(x).")
        if self.x.rank == 0:
            return self.x.sym_type  # scalar cannot be transposed
        if any_variadic(self.x.shape):
            ret_shape = get_new_variadic_symbol()
        else:
            ret_shape = x_shape[perm]
        return builtins.tensor(x_type, tuple(ret_shape))

    @precondition(allow=VALUE|SYMBOL)
    def value_inference(self):
        return np.transpose(self.x.val, axes=self.perm.val)


@register_op(doc_str="""
Rearranges elements in a tensor from depth (channel) into spatial dimensions.
Equivalent to PyTorch's pixel_shuffle.

Inputs

* x: <n, C x f^2, H, W, T> Required
    * Input tensor of rank 4
* upscale_factor: const<i32>
    * Factor to increase spatial resolution by

Outputs

* <n, C, H x f, W x f, T> where f is the upscale factor.

Type Domains

* T: f32
""")
class pixel_shuffle(Operation):
    input_spec = InputSpec(
        x=TensorInputType(),
        upscale_factor=IntInputType(const=True),
    )

    def __init__(self, **kwargs):
        super(pixel_shuffle, self).__init__(**kwargs)

    def type_inference(self):
        x_type = self.x.dtype
        n, c, h, w = self.x.shape
        f = self.upscale_factor.val
        ret_shape = (n, c // (f * f), h * f, w * f)
        return builtins.tensor(x_type, ret_shape)


@register_op(doc_str="""
Returns a tensor containing all windows of size, separated by stride along the given axis.

Inputs

* x: <*d0, d_axis, *dn, T>
    * Input tensor
* axis: const<i32>
    * Axis to perform the operation.
* size: const<i32>
    * Number of elements in the sliding window
* stride: const<i32>
    * The stride of the input elements in the sliding window
    * Optional, defaults to 1

Outputs

* <*d0, d_axis - size // stride + 1, size, *dn, T>
    * The output will be a tensor of rank N+1 where N is the input tensor rank

Type Domains

* T: f32
""")
class sliding_windows(Operation):
    input_spec = InputSpec(
        x=TensorInputType(),
        axis=IntInputType(const=True),
        size=IntInputType(const=True),
        stride=IntInputType(const=True, default=1)
    )

    def __init__(self, **kwargs):
        super(sliding_windows, self).__init__(**kwargs)

    def type_inference(self):
        x_shape = self.x.shape
        axis = self.axis.val
        size = self.size.val
        stride = self.stride.val
        ret_shape = list(x_shape)
        ret_shape[axis] = (x_shape[axis] - size) // stride + 1
        pos_axis = axis if axis >= 0 else axis + self.x.rank
        ret_shape.insert(pos_axis + 1, size)
        return builtins.tensor(self.x.dtype, tuple(ret_shape))
