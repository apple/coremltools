import six
import math
import numpy as np
import sympy as sm
import scipy
import operator
import functools

from coremltools.converters.nnv2.builtin_types.symbolic import (\
        is_symbolic, isscalar, is_symbolic, any_variadic, any_symbolic)
from coremltools.converters.nnv2.builtin_types.builtins.type_mapping import numpy_val_to_builtin_val

from coremltools.converters.nnv2.builtin_types import builtins
from ..program.program import (
        Operation, SsaBlock, Symbol,
        get_new_variadic_symbol, get_new_symbol)
from ..program.input_type import *
from .op_registry import register_op


def _broadcast_shapes(shape_x, shape_y):
    """
    Check and broadcast given input shapes.
    :param shape_x: tuple of int or symbols
        Shape of the first tensor (possibly symbolic).
    :param shape_y: tuple of int or symbols
        Shape of the second tensor (possibly symbolic).
    :return: tuple of int or symbols
        Result from broadcast.
    """
    if len(shape_x) < len(shape_y):
        shape_x = ([1] * (len(shape_y) - len(shape_x))) + shape_x
    if len(shape_y) < len(shape_x):
        shape_y = ([1] * (len(shape_x) - len(shape_y))) + shape_y

    ret_shapes = list()
    for i in range(len(shape_x)):
        x_unknown = is_symbolic(shape_x[i])
        y_unknown = is_symbolic(shape_y[i])
        if shape_x[i] == 1:
            ret_shapes.append(shape_y[i])
        elif shape_y[i] == 1:
            ret_shapes.append(shape_x[i])
        elif not y_unknown and shape_y[i] > 1:
            if not x_unknown and shape_x[i] != shape_y[i]:
                raise ValueError(
                    'Incompatible dim {} in shapes {} vs. {}'.format(
                        i, shape_x, shape_y))
            ret_shapes.append(shape_y[i])
        elif not x_unknown and shape_x[i] > 1:
            if not y_unknown and shape_x[i] != shape_y[i]:
                raise ValueError(
                    'Incompatible dim {} in shapes {} vs. {}'.format(
                        i, shape_x, shape_y))
            ret_shapes.append(shape_x[i])
        elif x_unknown or y_unknown:
            ret_shapes.append(sm.functions.Max(shape_x[i], shape_y[i]))
        else:
            assert (shape_x[i] == shape_y[i])
            ret_shapes.append(shape_x[i])

    return tuple(ret_shapes)


def _promoted_primitive_type(type1, type2):
    """
    Given a pair of tensor or primitive types, find the smallest type that can store an instance
    of their primitive type.
    """
    ptype1 = type1.get_primitive() if builtins.is_tensor(type1) else type1
    ptype2 = type2.get_primitive() if builtins.is_tensor(type2) else type2
    return builtins.promote_types(ptype1, ptype2)

# rdar://58622145
class elementwise_unary(Operation):
    input_types = InputSpec(
            x = ScalarOrTensorInputType(),
            )

    def __init__(self, **kwargs):
        super(elementwise_unary, self).__init__(**kwargs)

    def type_inference(self):
        return self.x.sym_type

class elementwise_binary(Operation):
    input_types = InputSpec(
            x = ScalarOrTensorInputType(),
            y = ScalarOrTensorInputType(),
            )

    def __init__(self, **kwargs):
        super(elementwise_binary, self).__init__(**kwargs)

    def type_inference(self):
        typea = self.x.sym_type
        typeb = self.y.sym_type
        primitive_type = _promoted_primitive_type(typea, typeb)
        if primitive_type is None:
            raise ValueError('Incompatible primitive types in broadcast operation')
        primitive_type = self.get_dtype(primitive_type)

        # broadcast
        if not builtins.is_tensor(typea) and not builtins.is_tensor(typeb):
            # both typea and typeb are not tensors
            return primitive_type
        if builtins.is_tensor(typea) and not builtins.is_tensor(typeb):
            # a is tensor, b is not
            return builtins.tensor(primitive_type, typea.get_shape())
        if not builtins.is_tensor(typea) and builtins.is_tensor(typeb):
            # a is not tensor, b is
            return builtins.tensor(primitive_type, typeb.get_shape())

        # both a, b are tensors
        shapea = list(typea.get_shape())
        shapeb = list(typeb.get_shape())
        ret_shape = _broadcast_shapes(shapea, shapeb)
        return builtins.tensor(primitive_type, ret_shape)

    def eval(self):
        return self.get_operator()(self.x.val, self.y.val)

    def get_operator(self):
        """
        All subclasses have to implement this.
        """
        raise NotImplementedError()

    def get_dtype(self, promoted_dtype):
        """
        Override if output primitive type is different from input types
        (e.g., less, greater)
        """
        return promoted_dtype


class Pooling(Operation):
    input_types = InputSpec(
        x=TensorInputType(),
        kernel_sizes=IntTensorInputType(const=True),
        strides=IntTensorInputType(const=True, optional=True),
        pad_type=StringInputType(const=True),
        pad=IntTensorInputType(const=True, optional=True),
    )

    def __init__(self, **kwargs):
        super(Pooling, self).__init__(**kwargs)

    def type_inference(self):
        ksize = self.kernel_sizes.val
        x_shape = self.x.shape
        D_in_rank = len(x_shape) - 2

        strides = [1] * D_in_rank if self.strides is None else self.strides.val
        pad_type = 'valid' if self.pad_type is None else self.pad_type.val.lower()
        pad = None if self.pad is None else self.pad.val
        D_in = x_shape[2:]  # spatial dimensions

        if pad_type == 'same':
            D_out_shape = [int(math.ceil(float(d) / float(s))) for d, s in zip(D_in, strides)]
        else:
            # rdar://59740053 (Padding Calculation for Conv2D does not work for custom padding)
            pad = _conv2d_pad(pad_type, D_in_rank, pad, ksize, strides)
            D_out_shape = [
                ((D_in[r] + pad[r] - (ksize[r] - 1) - 1) // strides[r] + 1) for r in range(D_in_rank)
            ]
        ret_shape = list(x_shape[:2]) + D_out_shape
        return builtins.tensor(self.x.dtype, tuple(ret_shape))


class RandomDistribution(Operation):
    input_types = InputSpec(
        shape=IntTensorInputType(),
    )

    def __init__(self, **kwargs):
        super(RandomDistribution, self).__init__(**kwargs)

    def type_inference(self):
        if any_symbolic(self.shape.shape):
            # We can't infer any shape if shape has variable length.
            return builtins.tensor(builtins.fp32, (get_new_variadic_symbol(),))

        # shape has fixed length here.
        if self.shape.sym_val is None:
            shape = tuple([get_new_symbol() for _ in range(self.shape.shape[0])])
            return builtins.tensor(builtins.fp32, shape)

        return builtins.tensor(builtins.fp32, tuple(self.shape.val.tolist()))


@register_op(doc_str='TODO')
class abs(elementwise_unary):
    def __init__(self, **kwargs):
        super(abs, self).__init__(**kwargs)

    def eval(self):
        return np.abs(self.x.val)

@register_op(doc_str='TODO')
class acos(elementwise_unary):
    def __init__(self, **kwargs):
        super(acos, self).__init__(**kwargs)

    def eval(self):
        return np.arccos(self.x.val)

@register_op(doc_str='TODO')
class asin(elementwise_unary):
    def __init__(self, **kwargs):
        super(asin, self).__init__(**kwargs)

    def eval(self):
        return np.arcsin(self.x.val)

@register_op(doc_str='TODO')
class atan(elementwise_unary):
    def __init__(self, **kwargs):
        super(atan, self).__init__(**kwargs)

    def eval(self):
        return np.arctan(self.x.val)


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
    input_types = InputSpec(
        x=TensorInputType(),
        lower=IntInputType(const=True, default=-1),
        upper=IntInputType(const=True, default=-1),
    )

    def __init__(self, **kwargs):
        super(band_part, self).__init__(**kwargs)

    def type_inference(self):
        return self.x.sym_type


@register_op(doc_str='TODO')
class atanh(elementwise_unary):
    def __init__(self, **kwargs):
        super(atanh, self).__init__(**kwargs)

    def eval(self):
        return np.arctanh(self.x.val)

@register_op(doc_str='TODO')
class ceil(elementwise_unary):
    def __init__(self, **kwargs):
        super(ceil, self).__init__(**kwargs)

    def eval(self):
        return np.ceil(self.x.val)

@register_op(doc_str='TODO')
class clip(Operation):
    input_types = InputSpec(
            x = ScalarOrTensorInputType(),
            alpha = FloatInputType(const=True),
            beta = FloatInputType(const=True),
            )

    def __init__(self, **kwargs):
        super(clip, self).__init__(**kwargs)

    def type_inference(self):
        return self.x.sym_type

    def eval(self):
        return np.minimum(np.maximum(self.x.val, self.alpha.val), self.beta.val)

@register_op(doc_str='TODO')
class cos(elementwise_unary):
    def __init__(self, **kwargs):
        super(cos, self).__init__(**kwargs)

    def eval(self):
        return np.cos(self.x.val)

@register_op(doc_str='TODO')
class cosh(elementwise_unary):
    def __init__(self, **kwargs):
        super(cosh, self).__init__(**kwargs)

    def eval(self):
        return np.cosh(self.x.val)

@register_op(doc_str='TODO')
class equal(elementwise_binary):
    def __init__(self, **kwargs):
        super(equal, self).__init__(**kwargs)

    def get_operator(self):
        return np.equal

    def get_dtype(self, promoted_dtype):
        return builtins.bool

@register_op(doc_str='TODO')
class exp(elementwise_unary):
    def __init__(self, **kwargs):
        super(exp, self).__init__(**kwargs)

    def eval(self):
        return np.exp(self.x.val)

@register_op(doc_str='TODO')
class exp2(elementwise_unary):
    def __init__(self, **kwargs):
        super(exp2, self).__init__(**kwargs)

    def eval(self):
        return np.exp2(self.x.val)


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
    input_types = InputSpec(
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

        return builtins.tensor(builtins.fp32, tuple(self.shape.val.tolist()))

    def eval(self):
        return np.full(shape=self.shape.val, fill_value=self.value.val)


@register_op(doc_str='TODO')
class floor(elementwise_unary):
    def __init__(self, **kwargs):
        super(floor, self).__init__(**kwargs)

    def eval(self):
        return np.floor(self.x.val)

@register_op(doc_str='TODO')
class floor_div(elementwise_binary):
    def __init__(self, **kwargs):
        super(floor_div, self).__init__(**kwargs)

    def get_operator(self):
        return operator.floordiv

@register_op(doc_str='TODO')
class greater(elementwise_binary):
    def __init__(self, **kwargs):
        super(greater, self).__init__(**kwargs)

    def get_operator(self):
        return operator.gt

    def get_dtype(self, promoted_dtype):
        return builtins.bool

@register_op(doc_str='TODO')
class greater_equal(elementwise_binary):
    def __init__(self, **kwargs):
        super(greater_equal, self).__init__(**kwargs)

    def get_operator(self):
        return operator.ge

    def get_dtype(self, promoted_dtype):
        return builtins.bool

@register_op(doc_str='TODO')
class less(elementwise_binary):
    def __init__(self, **kwargs):
        super(less, self).__init__(**kwargs)

    def get_operator(self):
        return operator.lt

    def get_dtype(self, promoted_dtype):
        return builtins.bool

@register_op(doc_str='TODO')
class less_equal(elementwise_binary):
    def __init__(self, **kwargs):
        super(less_equal, self).__init__(**kwargs)

    def get_operator(self):
        return operator.le

    def get_dtype(self, promoted_dtype):
        return builtins.bool

@register_op(doc_str='TODO')
class log(elementwise_unary):
    def __init__(self, **kwargs):
        super(log, self).__init__(**kwargs)

    def eval(self):
        return np.log(self.x.val)

@register_op(doc_str='TODO')
class logical_and(elementwise_binary):
    def __init__(self, **kwargs):
        super(logical_and, self).__init__(**kwargs)

    def get_operator(self):
        return np.logical_and

    def get_dtype(self, promoted_dtype):
        return builtins.bool

@register_op(doc_str='TODO')
class logical_not(elementwise_unary):
    def __init__(self, **kwargs):
        super(logical_not, self).__init__(**kwargs)

    def get_operator(self):
        return np.logical_not

    def get_dtype(self, promoted_dtype):
        return builtins.bool

@register_op(doc_str='TODO')
class logical_or(elementwise_binary):
    def __init__(self, **kwargs):
        super(logical_or, self).__init__(**kwargs)

    def get_operator(self):
        return np.logical_or

    def get_dtype(self, promoted_dtype):
        return builtins.bool

@register_op(doc_str='TODO')
class logical_xor(elementwise_binary):
    def __init__(self, **kwargs):
        super(logical_xor, self).__init__(**kwargs)

    def get_operator(self):
        return np.logical_xor

    def get_dtype(self, promoted_dtype):
        return builtins.bool

@register_op(doc_str='TODO')
class maximum(elementwise_binary):
    def __init__(self, **kwargs):
        super(maximum, self).__init__(**kwargs)

    def get_operator(self):
        return np.maximum

@register_op(doc_str='TODO')
class minimum(elementwise_binary):
    def __init__(self, **kwargs):
        super(minimum, self).__init__(**kwargs)

    def get_operator(self):
        return np.minimum

@register_op(doc_str='TODO')
class mod(elementwise_binary):
    def __init__(self, **kwargs):
        super(mod, self).__init__(**kwargs)

    def get_operator(self):
        return operator.mod

@register_op(doc_str='TODO')
class mul(elementwise_binary):
    def __init__(self, **kwargs):
        super(mul, self).__init__(**kwargs)

    def get_operator(self):
        return operator.mul


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
    input_types = InputSpec(
        x=TensorInputType()
    )

    def __init__(self, **kwargs):
        super(non_zero, self).__init__(**kwargs)

    def type_inference(self):
        shape = tuple([get_new_symbol(), self.x.rank])
        return builtins.tensor(self.x.dtype, shape)

    def eval(self):
        return np.transpose(np.nonzero(self.x.val))


@register_op(doc_str='TODO')
class not_equal(elementwise_binary):
    def __init__(self, **kwargs):
        super(not_equal, self).__init__(**kwargs)

    def get_operator(self):
        return operator.ne

    def get_dtype(self, promoted_dtype):
        return builtins.bool

@register_op(doc_str='TODO')
class real_div(elementwise_binary):
    def __init__(self, **kwargs):
        super(real_div, self).__init__(**kwargs)

    def get_operator(self):
        return operator.truediv

@register_op(doc_str='TODO')
class round(elementwise_unary):
    def __init__(self, **kwargs):
        super(round, self).__init__(**kwargs)

    def eval(self):
        return np.round(self.x.val)

@register_op(doc_str='TODO')
class rsqrt(elementwise_unary):
    def __init__(self, **kwargs):
        super(rsqrt, self).__init__(**kwargs)

    def eval(self):
        return 1. / np.sqrt(self.x.val)

@register_op(doc_str='TODO')
class sign(elementwise_unary):
    def __init__(self, **kwargs):
        super(sign, self).__init__(**kwargs)

    def eval(self):
        return np.sign(self.x.val)

@register_op(doc_str='TODO')
class sin(elementwise_unary):
    def __init__(self, **kwargs):
        super(sin, self).__init__(**kwargs)

    def eval(self):
        return np.sin(self.x.val)

@register_op(doc_str='TODO')
class sinh(elementwise_unary):
    def __init__(self, **kwargs):
        super(sinh, self).__init__(**kwargs)

    def eval(self):
        return np.sinh(self.x.val)


# rdar://58622145
@register_op(doc_str='TODO')
class avg_pool(Pooling):
    input_types = InputSpec(
        x=TensorInputType(),
        kernel_sizes=IntTensorInputType(const=True),
        strides=IntTensorInputType(const=True, optional=True),
        pad_type=StringInputType(const=True),
        pad=IntTensorInputType(const=True, optional=True),
        exclude_padding_from_average=BoolInputType(const=True, default=False)
    )

    def __init__(self, **kwargs):
        super(avg_pool, self).__init__(**kwargs)


# rdar://58622145
@register_op(doc_str='TODO')
class softmax(Operation):
    input_types = InputSpec(
            logit = TensorInputType(),
            axis = IntInputType(const=True, default=-1),
            )

    def __init__(self, **kwargs):
        super(softmax, self).__init__(**kwargs)

    def type_inference(self):
        return self.logit.sym_type

    def eval(self):
        x = self.logit.val
        axis = self.axis.val
        e_x = np.exp(x - np.amax(x, axis=axis))
        return e_x / e_x.sum()

@register_op(doc_str='TODO')
class pow(elementwise_binary):
    def __init__(self, **kwargs):
        super(pow, self).__init__(**kwargs)

    def get_operator(self):
        return operator.pow

# rdar://58622145
@register_op(doc_str='TODO')
class const(Operation):
    input_types = InputSpec(
            mode = InternalStringInputType(const=True,
                default="immediate_value"),
            val = InternalScalarOrTensorInputType(const=True),
            )

    def __init__(self, **kwargs):
        super(const, self).__init__(**kwargs)

    def type_inference(self):
        builtin_type, _ = self._get_type_val(self.val.val)
        return builtin_type

    def eval(self):
        _, val = self._get_type_val(self.val.val)
        return val

    def _get_type_val(self, value):

        if isinstance(value, float):
            value = np.float32(value)
        elif isinstance(value, bool):
            value = np.bool(value)
        elif isinstance(value, six.integer_types):
            value = np.int32(value)
        elif isinstance(value, (tuple, list)):
            value = np.array(value)
            if value.dtype == np.int64:
                # We use int32 by default.
                value = value.astype(np.int32)

        if not isinstance(value, (np.generic, np.ndarray,
            six.string_types, bool)):
            raise ValueError("Unknown value for constant: {}".format(value))

        _, builtin_type = numpy_val_to_builtin_val(value)
        return builtin_type, value


# Internal const can have symbolic value (for testing purpose)
@register_op(doc_str='TODO')
class _const_symbolic(const):
    def __init__(self, **kwargs):
        super(_const_symbolic, self).__init__(**kwargs)

    def type_inference(self):
        builtin_type, _ = self._get_type_val(self.val.sym_val)
        return builtin_type

    def sym_eval(self):
        # We allow symbolic values in _const_symbolic
        _, val = self._get_type_val(self.val.sym_val)
        return val


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
    input_types = InputSpec(
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


@register_op(doc_str="""
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
""")
class linear(Operation):
    input_types = InputSpec(
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
        return builtins.tensor(x_type, tuple(shape))

    def eval(self):
        res = np.matmul(self.x.val, np.transpose(self.weight.val))
        if self.bias is not None:
            res += self.bias.val
        return res


# rdar://58622145
@register_op(doc_str='TODO')
class matmul(Operation):
    input_types = InputSpec(
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
        if not (x_shape[-1] == y_shape[-2] or \
               is_symbolic(x_shape[-1]) or \
               is_symbolic(y_shape[-2])):
            msg = "Op {} (matmul): x {}, y {} are not broadcastable"
            raise ValueError(msg.format(self.name, self.x.shape, self.y.shape))

        ret_shape = list(_broadcast_shapes(x_shape[:-2], y_shape[:-2]))
        ret_shape += [x_shape[-2], y_shape[-1]]
        return builtins.tensor(x_type, tuple(ret_shape))

    def eval(self):
        x = self.x.val
        if self.transpose_x.val:
            x = np.transpose(x)
        y = self.y.val
        if self.transpose_y.val:
            y = np.transpose(y)
        return np.matmul(x, y)


def _conv2d_pad(pad_type, num_dims, custom_pad, filter_dims, strides):
    # pad = [t+b, l+r]
    if pad_type == 'same':
        return [d - 1 for d in filter_dims]
    if pad_type == 'valid':
        return [0] * 2
    if pad_type == 'custom':
        if custom_pad is None or len(custom_pad) != 2*num_dims:
            raise ValueError('Invalid custom_pad.')
        return custom_pad
    raise ValueError('Invalid padding pad_type "{}"'.format(pad_type))


# rdar://58622145
@register_op(doc_str='TODO')
class conv(Operation):
    input_types = InputSpec(
            x = TensorInputType(),
            W = TensorInputType(const=True),
            strides = TensorInputType(const=True, optional=True),
            pad_type = StringInputType(const=True),
            pad = IntTensorInputType(const=True, optional=True),
            dilations = TensorInputType(const=True, optional=True),
            group = IntInputType(const=True, default=1),
            B = TensorInputType(const=True, optional=True),
            )

    def __init__(self, **kwargs):
        super(conv, self).__init__(**kwargs)

    def type_inference(self):
        inshape = self.x.shape
        f_shape = self.W.shape
        kernel_shape = f_shape[2:]
        num_dims = len(inshape) - 2

        strides = [1]*num_dims if self.strides is None else self.strides.val
        dilations = [1]*num_dims if self.dilations is None else self.dilations.val
        pad = None if self.pad is None else self.pad.val
        N = inshape[0]
        C_out = f_shape[0]
        D_in = inshape[2:]  # spatial dimensions
        pad_type = self.pad_type.val
        if pad_type == 'same':
            for k in kernel_shape:
                if k % 2 == 0:
                    msg = "Even kernel size {} is disallowed " + \
                        "under same padding. Use custom padding instead"
                    raise ValueError(msg.format(kernel_shape))
        pad = _conv2d_pad(pad_type,
                num_dims, pad, kernel_shape, strides)

        D_out_shape = [
            int((D_in[r] + pad[r] - dilations[r] * (kernel_shape[r] - 1) - 1) \
                / strides[r] + 1) for r in range(num_dims) ]
        retshape = [N, C_out] + D_out_shape
        return builtins.tensor(self.x.dtype, tuple(retshape))

@register_op(doc_str='TODO')
class gru(Operation):
    input_types = InputSpec(
            x = TensorInputType(),
            initial_h = TensorInputType(),
            weight = TensorInputType(const=True),
            bias = TensorInputType(const=True, optional=True, default=None),
            direction = StringInputType(const=True, default="forward"),
            output_sequence = BoolInputType(const=True, default=False),
            activations = PyTupleInputType(const=True, default=("sigmoid", "tanh")),
            )

    def __init__(self, **kwargs):
        super(gru, self).__init__(**kwargs)

    def type_inference(self):
        if self.x.rank != 3:
            raise ValueError('Invalid input shape. Expecting Rank 3 input, got {}'.format(len(self.x.shape)))

        sequence_length, batch_size, input_size = self.x.shape

        if self.weight.rank != 2:
            raise ValueError('Invalid weight shape. Expecting Rank 2 input, got {}'.format(len(self.weight.shape)))

        input_hidden_size, hidden_dim = self.weight.shape
        hidden_size = input_hidden_size - input_size

        direction = self.direction.val
        valid_directions = {'forward', 'reverse'}
        if direction not in valid_directions:
            raise ValueError('Direction {} not supported. Supported directions: {}'.format(direction, valid_directions))

        dim_factor = 3
        if hidden_size != (hidden_dim // dim_factor):
            raise ValueError("Incorrect weight matrix: hidden dim size mismatch. \
                              Provided  {}. Expecting <b, 3*H>".format(self.weight.shape))

        out_seq_len = sequence_length if self.output_sequence.val else 1
        output_shape = [out_seq_len, batch_size, hidden_size]
        output_h_shape = [batch_size, hidden_size]
        return builtins.tensor(self.x.dtype, tuple(output_shape)), \
               builtins.tensor(self.x.dtype, tuple(output_h_shape))

@register_op(doc_str='TODO')
class lstm(Operation):
    # Setting clip threshold in range signed 32-bit Integer range
    CLIP_DEFAULT_VAL = 2147483647.0
    input_types = InputSpec(
            x = TensorInputType(),
            initial_h = TensorInputType(),
            initial_c = TensorInputType(),
            weight = TensorInputType(const=True),
            bias = TensorInputType(const=True, optional=True, default=None),
            direction = StringInputType(const=True, default="forward"),
            output_sequence = BoolInputType(const=True, default=False),
            activations = PyTupleInputType(const=True, default=("sigmoid", "tanh", "tanh")),
            peephole = TensorInputType(const=True, optional=True, default=None),
            clip = FloatInputType(const=True, default=CLIP_DEFAULT_VAL)
            )

    def __init__(self, **kwargs):
        super(lstm, self).__init__(**kwargs)

    def type_inference(self):
        if self.x.rank != 3:
            raise ValueError('Invalid input shape. Expecting Rank 3 input, got {}'.format(len(self.x.shape)))

        sequence_length, batch_size, input_size = self.x.shape

        if self.weight.rank != 2:
            raise ValueError('Invalid weight shape. Expecting Rank 2 input, got {}'.format(len(self.weight.shape)))

        input_hidden_size, hidden_dim = self.weight.shape
        hidden_size = input_hidden_size - input_size

        direction = self.direction.val
        valid_directions = {'forward', 'reverse', 'bidirectional'}
        if direction not in valid_directions:
            raise ValueError('Direction {} not supported. Supported directions: {}'.format(direction, valid_directions))

        dim_factor = 8 if direction == "bidirectional" else 4
        if hidden_size != (hidden_dim // dim_factor):
            raise ValueError("Incorrect weight matrix: hidden dim size mismatch. \
                              Provided  {}. Expecting <b, 4*DIRECTION*H>".format(self.weight.shape))

        out_seq_len = sequence_length if self.output_sequence.val else 1
        num_directions = dim_factor // 4
        output_shape = [out_seq_len, batch_size, num_directions*hidden_size]
        output_h_shape = [batch_size, num_directions*hidden_size]
        output_c_shape = [batch_size, num_directions*hidden_size]
        return builtins.tensor(self.x.dtype, tuple(output_shape)), \
               builtins.tensor(self.x.dtype, tuple(output_h_shape)), \
               builtins.tensor(self.x.dtype, tuple(output_c_shape))

# rdar://58622145
@register_op(doc_str='TODO')
class pad(Operation):
    input_types = InputSpec(
            x = TensorInputType(),
            pad = IntTensorInputType(const=True),
            mode = StringInputType(const=True, default="constant"),
            constant_val = FloatInputType(const=True, default=0.),
            )

    def __init__(self, **kwargs):
        super(pad, self).__init__(**kwargs)

        if self.pad.shape[0] % 2 != 0:
            raise ValueError("Padding must be even! Provided {}".format(self.pad.shape[0]))

        mode = self.mode.val
        if mode in {"reflect", "replicate"}:
            # Reflect and Replicate mode only support padding to last two dimensions
            if self.pad.shape[0] != 4:
                msg = "For {} mode, padding is only supported on last two dimension, provided" + \
                      " {}".format(mode, self.pad.val)
                raise ValueError("Incorrect Pad configuration! {}".format(msg))

    def type_inference(self):
        in_shape = self.x.shape
        pad = self.pad.val
        ret_shape = list(in_shape)
        pad_index = 0
        for i in range(len(ret_shape) - pad.shape[0] // 2, len(ret_shape)):
            ret_shape[i] = in_shape[i] + pad[2*pad_index] + pad[2*pad_index + 1]
            pad_index += 1
        return builtins.tensor(self.x.dtype, ret_shape)

    def eval(self):
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


# rdar://58622145
@register_op(doc_str='TODO')
class max_pool(Pooling):
    input_types = InputSpec(
        x=TensorInputType(),
        kernel_sizes=IntTensorInputType(const=True),
        strides=IntTensorInputType(const=True, optional=True),
        pad_type=StringInputType(const=True),
        pad=IntTensorInputType(const=True, optional=True),
    )

    def __init__(self, **kwargs):
        super(max_pool, self).__init__(**kwargs)


@register_op(doc_str='TODO')
class rnn(Operation):
    input_types = InputSpec(
            x = TensorInputType(),
            initial_h = TensorInputType(),
            weight = TensorInputType(const=True),
            bias = TensorInputType(const=True, optional=True, default=None),
            direction = StringInputType(const=True, default="forward"),
            output_sequence = BoolInputType(const=True, default=False),
            activation = StringInputType(const=True, default="tanh")
            )

    def __init__(self, **kwargs):
        super(rnn, self).__init__(**kwargs)

    def type_inference(self):
        if self.x.rank != 3:
            raise ValueError('Invalid input shape. Expecting Rank 3 input, got {}'.format(len(self.x.shape)))

        sequence_length, batch_size, input_size = self.x.shape

        if self.weight.rank != 2:
            raise ValueError('Invalid weight shape. Expecting Rank 2 input, got {}'.format(len(self.weight.shape)))

        _, hidden_size = self.weight.shape

        direction = self.direction.val
        valid_directions = {'forward', 'reverse'}
        if direction not in valid_directions:
            raise ValueError('Direction {} not supported. Supported directions: {}'.format(direction, valid_directions))

        out_seq_len = sequence_length if self.output_sequence.val else 1
        output_shape = [out_seq_len, batch_size, hidden_size]
        output_h_shape = [batch_size, hidden_size]
        return builtins.tensor(self.x.dtype, tuple(output_shape)), \
               builtins.tensor(self.x.dtype, tuple(output_h_shape))


# rdar://58622145
@register_op(doc_str='TODO')
class batch_norm(Operation):
    input_types = InputSpec(
        x=TensorInputType(),
        mean=TensorInputType(const=True),
        variance=TensorInputType(const=True),
        gamma=TensorInputType(const=True, optional=True),
        beta=TensorInputType(const=True, optional=True),
        epsilon=FloatInputType(const=True, default=1e-5),
    )

    def __init__(self, **kwargs):
        super(batch_norm, self).__init__(**kwargs)

    def type_inference(self):
        return self.x.sym_type


# rdar://58622145
@register_op(doc_str='TODO')
class add(elementwise_binary):
    def __init__(self, **kwargs):
        super(add, self).__init__(**kwargs)

    def get_operator(self):
        return operator.add

@register_op(doc_str='TODO')
class sqrt(elementwise_unary):
    def __init__(self, **kwargs):
        super(sqrt, self).__init__(**kwargs)

    def eval(self):
        return np.sqrt(self.x.val)

@register_op(doc_str='TODO')
class square(elementwise_unary):
    def __init__(self, **kwargs):
        super(square, self).__init__(**kwargs)

    def eval(self):
        return np.square(self.x.val)

# rdar://58622145
@register_op(doc_str='TODO')
class sub(elementwise_binary):
    def __init__(self, **kwargs):
        super(sub, self).__init__(**kwargs)

    def get_operator(self):
        return operator.sub


@register_op(doc_str="""
Returns elementwise applied rectified linear activation: min(x, 0)

Parameters
----------
x: <*, f32>, required

Returns
-------
<*, f32>, a tensor of the same shape as x.
""")
class relu(elementwise_unary):
    def __init__(self, **kwargs):
        super(relu, self).__init__(**kwargs)

    def eval(self):
        return np.maximum(self.x.val, 0)


@register_op(doc_str="""
Returns elementwise applied rectified linear activation: max(min(x, 0), 6)

Parameters
----------
x: <*, f32>, required

Returns
-------
<*, f32>, a tensor of the same shape as x.
""")
class relu6(elementwise_unary):
    def __init__(self, **kwargs):
        super(relu6, self).__init__(**kwargs)

    def eval(self):
        return np.minimum(np.maximum(self.x.val, 0), 6)


@register_op(doc_str="""
Returns the elementwise gaussian error linear unit activation on x.

Parameters
----------
x: <*, f32>, required

Returns
-------
<*, f32>, a tensor of the same shape as x.
""")
class gelu(elementwise_unary):
    def __init__(self, **kwargs):
      super(gelu, self).__init__(**kwargs)

    def eval(self):
        return 0.5 * self.x.val * (1 + scipy.special.erf(self.x.val / np.sqrt(2)))


@register_op(doc_str="""
Returns x if x >= alpha, 0 otherwise.

Parameters
----------
x: <*, f32>, required
alpha: const<f32>, optional, default = 1

Returns
-------
<*, f32>, a tensor of the same shape as x.
""")
class thresholded_relu(Operation):
    input_types = InputSpec(
            x = ScalarOrTensorInputType(),
            alpha = FloatInputType(const=True, default=1),
            )

    def __init__(self, **kwargs):
        super(thresholded_relu, self).__init__(**kwargs)

    def type_inference(self):
        return self.x.sym_type

    def eval(self):
        return np.maximum(self.x.val - self.alpha.val, 0)


@register_op(doc_str='TODO')
class tan(elementwise_unary):
    def __init__(self, **kwargs):
        super(tan, self).__init__(**kwargs)

    def eval(self):
        return np.tan(self.x.val)


@register_op(doc_str='TODO')
class threshold(Operation):
    input_types = InputSpec(
            x = ScalarOrTensorInputType(),
            alpha = FloatInputType(const=True),
            )

    def __init__(self, **kwargs):
        super(threshold, self).__init__(**kwargs)

    def type_inference(self):
        return self.x.sym_type

    def eval(self):
        return np.maximum(self.x.val, self.alpha.val)


@register_op(doc_str="""
Returns a tensor with specified shape with random values from a Bernoulli distribution.

.. math::

    f(k) = \begin{cases}1-p  &\text{if } k = 0\\
                        p    &\text{if } k = 1\end{cases}

for :math:`k` in :math:`\{0, 1\}`.

Parameters
----------
shape: <K, i32>, required
    Target output tensor shape.
    K is the rank of the output tensor. shape[k] > 0 for k = 0,..., K-1.
prob: const<f32>, optional
    The probability of sampling 1. Defaults to 0.5.
seed: const<i32>, optional
    Seed to create a reproducible sequence of values across multiple invokes.

Returns
-------
<*, T>, a tensor of given target output shape filled with random values.

See Also
--------
random_categorical, random_normal, random_uniform
""")
class random_bernoulli(RandomDistribution):
    input_types = InputSpec(
        shape=IntTensorInputType(),
        prob=FloatInputType(const=True, default=0.5),
        seed=IntInputType(const=True, default=-1)
    )

    def __init__(self, **kwargs):
        super(random_bernoulli, self).__init__(**kwargs)


@register_op(doc_str="""
Returns random values from a categorical distribution.

Parameters
----------
shape: <*D_in, T>
    N-dimensional tensor, one of logits (event log-probabilities) or probs
    (event probabilities). The first N - 1 dimensions specifies distributions,
    the last dimension represents a vector of probabilities.
mode: const<str>, optional
    One of ['logits', 'probs']. Defaults to 'logits'.
size: const<i32>, optional
    Number of samples to draw. Defaults to 1.
seed: const<i32>, optional
    Seed to create a reproducible sequence of values across multiple invokes.

Returns
-------
<*D_in[:-1] + [size], T>, a tensor of given target output shape filled with random values.

See Also
--------
random_bernoulli, random_normal, random_uniform
""")
class random_categorical(Operation):
    input_types = InputSpec(
        x=TensorInputType(),
        mode=StringInputType(const=True, default='logits'),
        size=IntInputType(const=True, default=1),
        seed=IntInputType(const=True, default=-1)
    )

    def __init__(self, **kwargs):
        super(random_categorical, self).__init__(**kwargs)

    def type_inference(self):
        output_shape = self.x.shape[:-1] + (self.size.val,)
        return builtins.tensor(builtins.fp32, output_shape)


@register_op(doc_str="""
Returns a tensor with specified shape with random values from a normal distribution.

.. math::

    f(x) = \frac{\exp(-x^2/2)}{\sqrt{2\pi}}

for a real number :math:`x`.

Parameters
----------
shape: <K, i32>, required
    Target output tensor shape.
    K is the rank of the output tensor. shape[k] > 0 for k = 0,..., K-1.
mean: const<f32>, optional
    The mean (center) of the normal distribution. Defaults to 0.0.
stddev: const<f32>, optional
    The standard deviation (width) of the normal distribution. Defaults to 1.0.
seed: const<i32>, optional
    Seed to create a reproducible sequence of values across multiple invokes.

Returns
-------
<*, T>, a tensor of given target output shape filled with random values.

See Also
--------
random_categorical, random_bernoulli, random_uniform
""")
class random_normal(RandomDistribution):
    input_types = InputSpec(
        shape=IntTensorInputType(),
        mean=FloatInputType(const=True, default=0.),
        stddev=FloatInputType(const=True, default=1.),
        seed=IntInputType(const=True, default=-1)
    )

    def __init__(self, **kwargs):
        super(random_normal, self).__init__(**kwargs)


@register_op(doc_str="""
Returns a tensor with specified shape with random values from a normal distribution.

.. math::

    p(x) = \frac{1}{high - low}

for a real number :math:`x`.

Parameters
----------
shape: <K, i32>, required
    Target output tensor shape.
    K is the rank of the output tensor. shape[k] > 0 for k = 0,..., K-1.
low: const<f32>, optional
    Lower boundary of the output interval (inclusive). Defaults to 0.0.
high: const<f32>, optional
    Upper boundary of the output interval (exclusive). Defaults to 1.0.
seed: const<i32>, optional
    Seed to create a reproducible sequence of values across multiple invokes.

Returns
-------
<*, T>, a tensor of given target output shape filled with random values.

See Also
--------
random_categorical, random_bernoulli, random_normal
""")
class random_uniform(RandomDistribution):
    input_types = InputSpec(
        shape=IntTensorInputType(),
        low=FloatInputType(const=True, default=0.),
        high=FloatInputType(const=True, default=1.),
        seed=IntInputType(const=True, default=-1)
    )

    def __init__(self, **kwargs):
        super(random_uniform, self).__init__(**kwargs)


# rdar://58622145
@register_op(doc_str='TODO')
class transpose(Operation):
    input_types = InputSpec(
            x = TensorInputType(),
            perm = IntTensorInputType(const=True),
            )

    def __init__(self, **kwargs):
        super(transpose, self).__init__(**kwargs)

    def type_inference(self):
        x_type = self.x.dtype
        x_shape = self.x.shape
        perm = self.perm.val
        x_shape = np.array(x_shape)
        if len(perm) != len(x_shape):
            raise ValueError("perm should have the same length as rank(x).")
        new_shape = x_shape[perm]
        return builtins.tensor(x_type, tuple(new_shape))

    def eval(self):
        return np.transpose(self.x.val, axes=self.perm.val)


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
    input_types = InputSpec(
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

    def eval(self):
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


# rdar://58622145
@register_op(doc_str='TODO')
class squeeze(Operation):
    input_types = InputSpec(
            x = TensorInputType(),
            axes = TensorInputType(const=True, optional=True),
            )

    def __init__(self, **kwargs):
        super(squeeze, self).__init__(**kwargs)

    def type_inference(self):
        x_type = self.x.dtype
        x_shape = self.x.shape
        squeezed_shape = list(x_shape)
        if self.axes is None:
            # Squeeze all single-dim
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

    def eval(self):
        return np.squeeze(self.x.val, axis=tuple(self.axes.val))


# We use markdown syntax. Italicizes (Required), (Optional...). Use e.g.,
# $\math{R}$ for math expression.
@register_op(doc_str= \
"""
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
    input_types = InputSpec(
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

    def eval(self):
        return np.expand_dims(self.x.val, axis=self.axis.val)


class ReductionAxes(Operation):

    input_types = InputSpec(
        x=TensorInputType(),
        axes=IntTensorInputType(const=True, default=None),
        keep_dims=BoolInputType(const=True, default=False),
    )

    def __init__(self, **kwargs):
        super(ReductionAxes, self).__init__(**kwargs)

    def type_inference(self):
        x_type = self.x.dtype
        x_shape = self.x.shape
        axes = self.axes.val if self.axes is not None else None
        if axes is None:
            axes = range(self.x.rank)
        keep_dims = self.keep_dims.val

        reduced_shape = list(x_shape)
        if keep_dims:
            for i in axes:
                reduced_shape[i] = 1
        else:
            # sort reverse so we can delete shape elements back to front
            axes = [axis if axis >= 0 else axis + len(reduced_shape) for axis in axes]
            for i in sorted(axes)[::-1]:
                reduced_shape.pop(i)
        if len(reduced_shape) == 0:
            return x_type  # scalar

        return builtins.tensor(x_type, tuple(reduced_shape))

    def eval(self):
        axes = tuple(self.axes.val) if self.axes is not None else None
        return self.get_operator()(self.x.val, axis=axes, keepdims=self.keep_dims.val)

    def get_operator(self):
        raise NotImplementedError()


class ReductionAxis(Operation):

    input_types = InputSpec(
        x=TensorInputType(),
        axis=IntInputType(const=True, optional=True),
        keep_dims=BoolInputType(const=True, default=False),
    )

    def __init__(self, **kwargs):
        super(ReductionAxis, self).__init__(**kwargs)

    def type_inference(self):
        x_type = self.x.dtype
        x_shape = self.x.shape
        axis = self.axis.val

        reduced_shape = list(x_shape)
        axis = axis if axis >= 0 else axis + len(reduced_shape)
        if self.keep_dims.val:
            reduced_shape[axis] = 1
        else:
            reduced_shape.pop(axis)

        return builtins.tensor(x_type, tuple(reduced_shape))

    def eval(self):
        return self.get_operator()(self.x.val, axis=self.axis.val)

    def get_operator(self):
        raise NotImplementedError()


# rdar://58622145
@register_op(doc_str='TODO')
class reduce_argmax(ReductionAxis):

    def __init__(self, **kwargs):
        super(reduce_argmax, self).__init__(**kwargs)

    def get_operator(self):
        return np.argmax


# rdar://58622145
@register_op(doc_str='TODO')
class reduce_argmin(ReductionAxis):

    def __init__(self, **kwargs):
        super(reduce_argmin, self).__init__(**kwargs)

    def get_operator(self):
        return np.argmin


# rdar://58622145
@register_op(doc_str='TODO')
class reduce_l1_norm(ReductionAxes):
    def __init__(self, **kwargs):
        super(reduce_l1_norm, self).__init__(**kwargs)

    def get_operator(self):
        def l1_norm(x, axis=None, keepdims=False):
            return np.sum(np.abs(x), axis=axis, keepdims=keepdims)

        return l1_norm


# rdar://58622145
@register_op(doc_str='TODO')
class reduce_l2_norm(ReductionAxes):
    def __init__(self, **kwargs):
        super(reduce_l2_norm, self).__init__(**kwargs)

    def get_operator(self):
        def l2_norm(x, axis=None, keepdims=False):
            return np.sqrt(np.sum(np.square(x), axis=axis, keepdims=keepdims))

        return l2_norm


# rdar://58622145
@register_op(doc_str='TODO')
class reduce_log_sum(ReductionAxes):
    def __init__(self, **kwargs):
        super(reduce_log_sum, self).__init__(**kwargs)

    def get_operator(self):
        def log_sum(x, axis=None, keepdims=False):
            return np.log(np.sum(x, axis=axis, keepdims=keepdims))

        return log_sum


# rdar://58622145
@register_op(doc_str='TODO')
class reduce_log_sum_exp(ReductionAxes):
    def __init__(self, **kwargs):
        super(reduce_log_sum_exp, self).__init__(**kwargs)

    def get_operator(self):
        return scipy.special.logsumexp


# rdar://58622145
@register_op(doc_str='TODO')
class reduce_max(ReductionAxes):
    def __init__(self, **kwargs):
        super(reduce_max, self).__init__(**kwargs)

    def get_operator(self):
        return np.max


# rdar://58622145
@register_op(doc_str='TODO')
class reduce_mean(ReductionAxes):
    def __init__(self, **kwargs):
        super(reduce_mean, self).__init__(**kwargs)

    def get_operator(self):
        return np.mean


# rdar://58622145
@register_op(doc_str='TODO')
class reduce_min(ReductionAxes):
    def __init__(self, **kwargs):
        super(reduce_min, self).__init__(**kwargs)

    def get_operator(self):
        return np.min


# rdar://58622145
@register_op(doc_str='TODO')
class reduce_prod(ReductionAxes):
    def __init__(self, **kwargs):
        super(reduce_prod, self).__init__(**kwargs)

    def get_operator(self):
        return np.prod


# rdar://58622145
@register_op(doc_str='TODO')
class reduce_sum(ReductionAxes):
    def __init__(self, **kwargs):
        super(reduce_sum, self).__init__(**kwargs)

    def get_operator(self):
        return np.sum


# rdar://58622145
@register_op(doc_str='TODO')
class reduce_sum_square(ReductionAxes):
    def __init__(self, **kwargs):
        super(reduce_sum_square, self).__init__(**kwargs)

    def get_operator(self):
        def sum_squre(x, axis=None, keepdims=False):
            return np.sum(np.square(x), axis=axis, keepdims=keepdims)

        return sum_squre


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
    input_types = InputSpec(
        x=TensorInputType(),
        axes=IntTensorInputType(const=True, optional=True),
    )

    def __init__(self, **kwargs):
        super(reverse, self).__init__(**kwargs)

    def type_inference(self):
        return self.x.sym_type

    def eval(self):
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
    input_types = InputSpec(
        x=TensorInputType(),
        lengths=IntTensorInputType(),
        seq_axis=IntInputType(const=True, default=0),
        batch_axis=IntInputType(const=True, default=0)
    )

    def __init__(self, **kwargs):
        super(reverse_sequence, self).__init__(**kwargs)

    def type_inference(self):
        return self.x.sym_type

    def eval(self):
        raise NotImplementedError('TODO')


# rdar://58622145
@register_op(doc_str='TODO')
class _make_tuple(Operation):
    input_types = InputSpec(
            elems = PyTupleInputType(),
            )

    def __init__(self, **kwargs):
        super(_make_tuple, self).__init__(**kwargs)

    def type_inference(self):
        # Tuple's type and value are derived directly from element values.
        return builtins.tuple(tuple(e.sym_type for e in self.elems.val))

    def eval(self):
        # self.elems.val is python tuple[Var]
        return self.elems.val


# rdar://58622145
@register_op(doc_str='TODO')
class while_loop(Operation):
    input_types = InputSpec(
            # arg name with underscore prefix won't be printed.
            _cond = PyFunctionInputType(),
            _body = PyFunctionInputType(),
            loop_vars = TupleInputType(),
            )

    def __init__(self, **kwargs):
        super(while_loop, self).__init__(**kwargs)

    def type_inference(self):
        # self.loop_vars is TupleVar
        # Cond block
        cond_block_name = self.name + '_cond'
        # SsaBlock takes a python tuple[Var]
        py_tuple_loop_vars = self.loop_vars.py_tuple()
        with SsaBlock(block_inputs=py_tuple_loop_vars, outer_op=self,
                name=cond_block_name) as cond_block:
            cond_func = self._cond.val
            cond_var = cond_func(cond_block.inputs)
            cond_block.set_outputs([cond_var])
            self.blocks.append(cond_block)
        if not isinstance(cond_var, Var) or cond_var.dtype != builtins.bool:
            msg = "Cond in while_loop {} should return bool, but got {}"
            raise ValueError(msg.format(self.name, cond_var.sym_type))

        # Body block
        body_block_name = self.name + '_body'
        with SsaBlock(block_inputs=py_tuple_loop_vars, outer_op=self,
                name=body_block_name) as body_block:
            body_func = self._body.val
            exit_vars = body_func(body_block.inputs)
            body_block.set_outputs(exit_vars)
            self.blocks.append(body_block)

        # Verify exit_vars has the same types as loop_vars
        for v_in, v_out in zip(py_tuple_loop_vars, exit_vars):
            if v_in.sym_type != v_out.sym_type:
                msg = "loop_vars {} changes in the body while_loop {}"
                raise ValueError(msg.format(v_in.name, self.name))

        # Cannot perform AUTO_VAL for while_loop
        return tuple((self.name + '_' + v.name, v.sym_type, None)
            for v in exit_vars)


# rdar://58622145
@register_op(doc_str='TODO')
class cond(Operation):
    input_types = InputSpec(
            pred = BoolInputType(),
            _true_fn = PyFunctionInputType(),
            _false_fn = PyFunctionInputType(),
            )

    def __init__(self, **kwargs):
        super(cond, self).__init__(**kwargs)

    def type_inference(self):
        # self.input_vars is TupleVar
        # Cond block
        true_block_name = self.name + '_true'
        with SsaBlock(name=true_block_name, outer_op=self) as true_block:
            true_func = self._true_fn.val
            true_ret_vars = true_func()
            if not isinstance(true_ret_vars, (tuple, list)):
                true_ret_vars = [true_ret_vars]
            true_block.set_outputs(true_ret_vars)
            self.blocks.append(true_block)

        false_block_name = self.name + '_false'
        with SsaBlock(name=false_block_name, outer_op=self) as false_block:
            false_func = self._false_fn.val
            false_ret_vars = false_func()
            if not isinstance(false_ret_vars, (tuple, list)):
                false_ret_vars = [false_ret_vars]
            false_block.set_outputs(false_ret_vars)
            self.blocks.append(false_block)

        # Verify true_ret_vars has the same types as false_ret_vars
        for i, (vt, vf) in enumerate(zip(true_ret_vars, false_ret_vars)):
            if vt.sym_type != vf.sym_type:
                msg = "true branch output {} type {} mismatch false branch" +\
                    " output type {}"
                import pdb
                pdb.set_trace()
                raise ValueError(msg.format(vt.name, vt.sym_type, vf.sym_type))

        # Cannot perform AUTO_VAL for cond
        return tuple((self.name + '_' + v.name, v.sym_type, None)
            for v in true_ret_vars)


@register_op(doc_str="""
Returns the gauss error function, applied elementwise to x.

Parameters
----------
x: <*, f32>, required

Returns
-------
<*, f32>, a tensor of the same shape as x.
""")
class erf(elementwise_unary):
    def __init__(self, **kwargs):
        super(erf, self).__init__(**kwargs)

    def eval(self):
        return scipy.special.erf(self.x.val)


@register_op(doc_str="""
Returns the elements selected from either a or b, depending on the cond.
Shape of cond, a, b must be broadcastable.

Inputs

* cond <*, T>
    * Tensor, when True (non-zero), select element from x, otherwise, y
* a <*, T> Optional
    * Tensor, values selected at indices where condition is True
    * Defaults to None.
* b <*, T> Optional
    * Tensor, values selected at indices where condition is False
    * Defaults to None.

Outputs

* <*, T>
    *  A tensor of shape equal to the broadcasted shape.

Type Domains

* T: f32
""")
class select(Operation):
    input_types = InputSpec(
        cond=TensorInputType(),
        a=TensorInputType(),
        b=TensorInputType()
    )

    def __init__(self, **kwargs):
        super(select, self).__init__(**kwargs)

    def type_inference(self):
        a_type = self.a.sym_type
        b_type = self.b.sym_type
        if all([a_type, b_type]):
            compatible, ret_type = builtins.is_tensor_and_is_compatible_general_shape(
                a_type, b_type
            )
            if compatible:
                return ret_type
            elif a_type == b_type:
                return a_type
            else:
                raise ValueError('Type mismatch {} vs. {}'.format(a_type, b_type))
        return a_type if a_type is not None else b_type

    def eval(self):
        return np.where(self.cond.val, self.a.val, self.b.val)


@register_op(doc_str="""
Returns sigmoid(x) element-wise.

Parameters
----------
x: <*, f32>, required

Returns
-------
<*, f32>, a tensor of the same shape as x.
""")
class sigmoid(elementwise_unary):
    def __init__(self, **kwargs):
        super(sigmoid, self).__init__(**kwargs)

    def eval(self):
        return 1/(1 + np.exp(-self.x.val))


@register_op(doc_str="""
Returns log( 1 + e^x ) elementwise.

Parameters
----------
x: <*, f32>, required

Returns
-------
<*, f32>, a tensor of the same shape as x.
""")
class softplus(elementwise_unary):
     def __init__(self, **kwargs):
         super(softplus, self).__init__(**kwargs)

     def eval(self):
         return np.log(1 + np.exp(-np.abs(self.x.val))) + np.maximum(self.x.val, 0)


@register_op(doc_str="""
Returns x / ( 1 + |x| ) applied elementwise.

Parameters
----------
x: <*, f32>, required

Returns
-------
<*, f32>, a tensor of the same shape as x.
""")
class softsign(elementwise_unary):
    def __init__(self, **kwargs):
        super(softsign, self).__init__(**kwargs)

    def eval(self):
        return self.x.val / (1 + np.abs(self.x.val))


@register_op(doc_str="""
Returns tanh applied elementwise.

Parameters
----------
x: <*, f32>, required

Returns
-------
<*, f32>, a tensor of the same shape as x.
""")
class tanh(elementwise_unary):
    def __init__(self, **kwargs):
        super(tanh, self).__init__(**kwargs)

    def eval(self):
        return np.tanh(self.x.val)


@register_op(doc_str="""
Returns elementwise min(beta, x) if x >= 0, min(beta, alpha * x) otherwise.

Parameters
----------
x: <*, f32>, required
alpha: const<f32>, required
beta: const<f32>, required

Returns
-------
<*, f32>, a tensor of the same shape as x.
""")
class clamped_relu(Operation):
    input_types = InputSpec(
            x = ScalarOrTensorInputType(),
            alpha = FloatInputType(const=True),
            beta = FloatInputType(const=True),
            )

    def __init__(self, **kwargs):
        super(clamped_relu, self).__init__(**kwargs)

    def eval(self):
        x = np.minimum(np.maximum(self.x.val, 0), self.beta.val)
        y = np.minimum(np.minimum(self.x.val, 0) * self.alpha.val, self.beta.val)
        return x + y

    def type_inference(self):
        return self.x.sym_type


@register_op(doc_str="""
Returns x if x > 0,  alpha * e^(x - 1) otherwise.

Parameters
----------
x: <*, f32>, required
alpha: const<f32>, optional, default = 1

Returns
-------
<*, f32>, a tensor of the same shape as x.
""")
class elu(Operation):
    input_types = InputSpec(
            x = ScalarOrTensorInputType(),
            alpha = FloatInputType(const=True, default=1),
            )

    def __init__(self, **kwargs):
        super(elu, self).__init__(**kwargs)

    def eval(self):
        b = np.copy(self.x.val)
        b[b < 0] = self.alpha.val * (np.exp(b[b < 0]) - 1)
        return b

    def type_inference(self):
        return self.x.sym_type


@register_op(doc_str='TODO')
class l2_pool(Pooling):
    input_types = InputSpec(
        x=TensorInputType(),
        kernel_sizes=IntTensorInputType(const=True),
        strides=IntTensorInputType(const=True, optional=True),
        pad_type=StringInputType(const=True),
        pad=IntTensorInputType(const=True, optional=True),
    )

    def __init__(self, **kwargs):
        super(l2_pool, self).__init__(**kwargs)


@register_op(doc_str="""
Elementwise apply x if x >= 0 else alpha * x

Parameters
----------
x: <*, f32>, required
alpha: const<f32>, optional, default= 0.01

Returns
-------
<*, f32>, a tensor of the same shape as x.
""")
class leaky_relu(Operation):
    input_types = InputSpec(
            x = ScalarOrTensorInputType(),
            alpha = FloatInputType(const=True, default=0.01),
        )

    def __init__(self, **kwargs):
        super(leaky_relu, self).__init__(**kwargs)

    def eval(self):
        b = np.copy(self.x.val)
        b[b < 0] *= self.alpha.val
        return b

    def type_inference(self):
        return self.x.sym_type


@register_op(doc_str="""
Applies elementwise x * alpha + beta.

Parameters
----------
x: <*, f32>, required
alpha: const<f32>, required
beta: const<f32>, optional, default = 0

Returns
-------
<*, f32>, a tensor of the same shape as x.
""")
class linear_activation(Operation):
    input_types = InputSpec(
            x = ScalarOrTensorInputType(),
            alpha = FloatInputType(const=True),
            beta = FloatInputType(const=True, default=0),
            )

    def __init__(self, **kwargs):
        super(linear_activation, self).__init__(**kwargs)

    def eval(self):
        return self.alpha.val * self.x.val + self.beta.val

    def type_inference(self):
        return self.x.sym_type


@register_op(doc_str="""
Returns alpha * tan(beta * x) element-wise. Input range is (-inf, inf).

Parameters
----------
x: <*, f32>, required
alpha: const<f32>, optional, default = 1
beta: const<f32>, optional, default = 1

Returns
-------
<*, f32>, a tensor of the same shape as x.
""")
class scaled_tanh(Operation):
    input_types = InputSpec(
            x = ScalarOrTensorInputType(),
            alpha = FloatInputType(const=True, default=1),
            beta = FloatInputType(const=True, default=1),
            )

    def __init__(self, **kwargs):
        super(scaled_tanh, self).__init__(**kwargs)

    def eval(self):
        return self.alpha.val * np.tanh(self.x.val * self.beta.val)

    def type_inference(self):
        return self.x.sym_type


@register_op(doc_str="""
Returns min( max( alpha * x + beta, 0 ), 1 ) elementwise.

Parameters
----------
x: <*, f32>, required
alpha: const<f32>, optional, default 0.2
beta: const<f32>, optional, default 0.5

Returns
-------
<*, f32>, a tensor of the same shape as x.
""")
class sigmoid_hard(Operation):
    input_types = InputSpec(
            x = ScalarOrTensorInputType(),
            alpha = FloatInputType(const=True, default=0.2),
            beta = FloatInputType(const=True, default=0.5),
            )

    def __init__(self, **kwargs):
        super(sigmoid_hard, self).__init__(**kwargs)

    def eval(self):
        return np.minimum(np.maximum((self.alpha.val * self.x.val) + self.beta.val, 0), 1)

    def type_inference(self):
        return self.x.sym_type


@register_op(doc_str="""
Returns x_i if x_i > 0, alpha_i * x_i otherwise, where i = 1 ... C

Parameters
----------
x: <*, C, n, m, f32>, required
alpha: const<C, f32>, required

Returns
-------
<*, f32>, a tensor of the same shape as x.
""")
class prelu(Operation):
    input_types = InputSpec(
            x = TensorInputType(),
            alpha = TensorInputType(const=True),
            )

    def __init__(self, **kwargs):
        super(prelu, self).__init__(**kwargs)

    def eval(self):
        alpha_br = self.alpha.val
        for i in range(1, len(self.x.shape)):
            alpha_br = np.expand_dims(alpha_br, i)
        x_pos = np.maximum(self.x.val, 0)
        b = np.minimum(self.x.val, 0)
        return x_pos + b * alpha_br

    def type_inference(self):
        if (len(self.x.shape) < 3):
            raise ValueError("x should be at least rank 3")
        if (len(self.alpha.val.shape) != 1):
            raise ValueError("alpha should be rank 1")
        if (self.x.shape[-3] != self.alpha.val.shape[0]):
            raise ValueError("Size of dimension 0 of alpha should be the same as " +
                             "the size of dimension -3 of x.")
        return self.x.sym_type


@register_op(doc_str="""
Returns alpha_i * log( 1 + e^( beta_i * x_i ) ), where i = 1 ... C

Parameters
----------
x: <*, C, n, m, f32>, required
alpha: const<C, f32>, required
beta: const<C, f32>, required

Returns
-------
<*, f32>, a tensor of the same shape as x.
""")
class softplus_parametric(Operation):
    input_types = InputSpec(
            x = TensorInputType(),
            alpha = TensorInputType(const=True),
            beta = TensorInputType(const=True),
            )

    def __init__(self, **kwargs):
        super(softplus_parametric, self).__init__(**kwargs)

    def eval(self):
        alpha_br = np.copy(self.alpha.val)
        beta_br = np.copy(self.beta.val)
        for i in range(1, len(self.x.val.shape)):
            alpha_br = np.expand_dims(alpha_br, i)
            beta_br = np.expand_dims(beta_br, i)
        return alpha_br * np.log(1 + np.exp(self.x.val * beta_br))

    def type_inference(self):
        if (len(self.x.shape) < 3):
            raise ValueError("x should be at least rank 3")
        if (len(self.alpha.val.shape) != 1):
            raise ValueError("alpha should be rank 1")
        if (self.x.shape[-3] != self.alpha.val.shape[0]):
            raise ValueError("Size of dimension 0 of alpha should be the same as " +
                             "the size of dimension -3 of x.")
        if (len(self.beta.val.shape) != 1):
            raise ValueError("beta should be rank 1")
        if (self.x.shape[-3] != self.beta.val.shape[0]):
            raise ValueError("Size of dimension 0 of beta should be the same as " +
                             "the size of dimension -3 of x.")
        return self.x.sym_type


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
    input_types = InputSpec(
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


# rdar://59195036
@register_op(doc_str='TODO')
class cumsum(Operation):
    input_types = InputSpec(
            x = TensorInputType(),
            axis = IntInputType(const=True, default=0),
            exclusive = BoolInputType(const=True, default=False),
            reverse = BoolInputType(const=True, default=False)
            )

    def __init__(self, **kwargs):
        super(cumsum, self).__init__(**kwargs)

    def eval(self):
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

# rdar://59195036
@register_op(doc_str="TODO")
class gather(Operation):
    input_types = InputSpec(
            x = TensorInputType(),
            indices = IntTensorInputType(),
            axis = IntInputType(const=True, default=0)
            )

    def __init__(self, **kwargs):
        super(gather, self).__init__(**kwargs)

    def eval(self):
        x = self.x.val
        indices = self.indices.val
        axis = self.axis.val
        return np.take(x, indices, axis)

    def type_inference(self):
        out_type = self.x.dtype

        if self.axis.val < -self.x.rank \
                or self.axis.val >= self.x.rank:
            raise IndexError(
                'Axis value {} is out of bounds for {} node {}'.format(
                    self.axis.val, self.op_type, self.name))

        axis = self.axis.val
        axis = axis if axis >= 0 else axis + self.x.rank
        out_shape = self.x.shape[:axis] + self.indices.shape + self.x.shape[axis + 1:]
        return builtins.tensor(out_type, out_shape)

# rdar://59195036
@register_op(doc_str="TODO")
class scatter(Operation):
    input_types = InputSpec(
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

# rdar://59195036
@register_op(doc_str="TODO")
class gather_along_axis(Operation):
    input_types = InputSpec(
            x = TensorInputType(),
            indices = IntTensorInputType(),
            axis = IntInputType(const=True, default=0)
            )

    def __init__(self, **kwargs):
        super(gather_along_axis, self).__init__(**kwargs)

    def eval(self):
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

        return builtins.tensor(self.x.dtype, self.indices.shape)

# rdar://59195036
@register_op(doc_str="TODO")
class scatter_along_axis(Operation):
    input_types = InputSpec(
            data = TensorInputType(),
            indices = IntTensorInputType(),
            updates = TensorInputType(),
            axis = IntInputType(const=True, default=0),
            mode = StringInputType(const=True, default="add")
            )

    def __init__(self, **kwargs):
        super(scatter_along_axis, self).__init__(**kwargs)

    def eval(self):
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

# rdar://59195036
@register_op(doc_str="TODO")
class gather_nd(Operation):
    input_types = InputSpec(
            x = TensorInputType(),
            indices = IntTensorInputType(),
            )

    def __init__(self, **kwargs):
        super(gather_nd, self).__init__(**kwargs)

    def type_inference(self):
        assert self.indices.shape[-1] <= self.x.rank
        out_type = self.x.dtype
        out_shape = self.indices.shape[:-1] + self.x.shape[self.indices.shape[-1]:]
        return builtins.tensor(out_type, out_shape)

# rdar://59195036
@register_op(doc_str="TODO")
class scatter_nd(Operation):
    input_types = InputSpec(
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
        assert self.updates.shape == tuple(expected_updates_shape)
        return self.data.sym_type

# rdar://58622145
@register_op(doc_str='TODO')
class tile(Operation):
    input_types = InputSpec(
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

    def eval(self):
        return np.tile(self.x.val, reps=self.reps.val)

@register_op(doc_str='TODO')
class instance_norm(Operation):
    input_types = InputSpec(
        x=TensorInputType(),
        gamma=TensorInputType(const=True, optional=True),
        beta=TensorInputType(const=True, optional=True),
        epsilon=FloatInputType(const=True, default=1e-5),
    )

    def __init__(self, **kwargs):
        super(instance_norm, self).__init__(**kwargs)

    def type_inference(self):
        return self.x.sym_type


@register_op(doc_str='TODO')
class l2_norm(Operation):
    input_types = InputSpec(
        x=TensorInputType(),
        axes=IntTensorInputType(),
        epsilon=FloatInputType(const=True, default=1e-12)
    )

    def __init__(self, **kwargs):
        super(l2_norm, self).__init__(**kwargs)

    def type_inference(self):
        return self.x.sym_type


@register_op(doc_str='TODO')
class layer_norm(Operation):
    input_types = InputSpec(
        x=TensorInputType(),
        axes=IntTensorInputType(const=True, optional=True),
        gamma=TensorInputType(const=True, optional=True),
        beta=TensorInputType(const=True, optional=True),
        epsilon=FloatInputType(const=True, default=1e-5),
    )

    def __init__(self, **kwargs):
        super(layer_norm, self).__init__(**kwargs)

    def type_inference(self):
        return self.x.sym_type

    def eval(self):
        def np_layer_norm(x, axes, gamma, beta, epsilon=1e-5):
            normalized_shape = x.shape[-len(axes):]
            gamma = np.ones(shape=normalized_shape) if gamma is None else gamma
            beta = np.zeros(shape=normalized_shape) if beta is None else beta
            num = x - np.mean(x, axis=tuple(axes), keepdims=True)
            dem = np.sqrt(
                np.sum(np.square(num), axis=tuple(axes), keepdims=True) /
                np.prod(normalized_shape) + epsilon)
            return num / dem * gamma + beta

        _axes = self.x.shape if self.axes is None else self.axes.val
        _gamma = None if self.gamma is None else self.gamma.val
        _beta = None if self.beta is None else self.beta.val
        return np_layer_norm(self.x.val, _axes, _gamma, _beta, self.epsilon.val)


@register_op(doc_str='TODO')
class local_response_norm(Operation):
    input_types = InputSpec(
        x=TensorInputType(),
        size=IntInputType(const=True),
        alpha=FloatInputType(const=True, default=1e-4),
        beta=FloatInputType(const=True, default=0.75),
        k=FloatInputType(const=True, default=1.0),
    )

    def __init__(self, **kwargs):
        super(local_response_norm, self).__init__(**kwargs)

    def type_inference(self):
        return self.x.sym_type


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
    input_types = InputSpec(
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

    def eval(self):
        indices = np.argsort(self.x.val, axis=self.axis.val)
        if not self.ascending.val:
            indices = np.argsort(-self.x.val, axis=self.axis.val)
        slc = [slice(None)] * self.x.rank
        slc[self.axis.val] = slice(0, self.k.val)
        indices = indices[tuple(slc)]
        values = np.take_along_axis(self.x.val, indices, axis=self.axis.val)
        return values, indices
