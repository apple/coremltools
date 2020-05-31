import scipy
from ._op_reqs import *

"""
Elementwise Unary Op Superclass
"""
class elementwise_unary(Operation):
    input_spec = InputSpec(
            x = ScalarOrTensorInputType(),
            )

    def __init__(self, **kwargs):
        super(elementwise_unary, self).__init__(**kwargs)

    def type_inference(self):
        return self.x.sym_type


"""
Elementwise unary op implmentation(s)
"""

@register_op(doc_str='TODO')
class abs(elementwise_unary):
    def __init__(self, **kwargs):
        super(abs, self).__init__(**kwargs)

    @precondition(allow=VALUE)
    def value_inference(self):
        return np.abs(self.x.val)


@register_op(doc_str='TODO')
class acos(elementwise_unary):
    def __init__(self, **kwargs):
        super(acos, self).__init__(**kwargs)

    @precondition(allow=VALUE)
    def value_inference(self):
        return np.arccos(self.x.val)


@register_op(doc_str='TODO')
class asin(elementwise_unary):
    def __init__(self, **kwargs):
        super(asin, self).__init__(**kwargs)

    @precondition(allow=VALUE)
    def value_inference(self):
        return np.arcsin(self.x.val)


@register_op(doc_str='TODO')
class atan(elementwise_unary):
    def __init__(self, **kwargs):
        super(atan, self).__init__(**kwargs)

    @precondition(allow=VALUE)
    def value_inference(self):
        return np.arctan(self.x.val)


@register_op(doc_str='TODO')
class atanh(elementwise_unary):
    def __init__(self, **kwargs):
        super(atanh, self).__init__(**kwargs)

    @precondition(allow=VALUE)
    def value_inference(self):
        return np.arctanh(self.x.val)


@register_op(doc_str='TODO')
class ceil(elementwise_unary):
    def __init__(self, **kwargs):
        super(ceil, self).__init__(**kwargs)

    @precondition(allow=VALUE)
    def value_inference(self):
        return np.ceil(self.x.val)


@register_op(doc_str='TODO')
class clip(Operation):
    input_spec = InputSpec(
            x = ScalarOrTensorInputType(),
            alpha = FloatInputType(const=True),
            beta = FloatInputType(const=True),
            )

    def __init__(self, **kwargs):
        super(clip, self).__init__(**kwargs)

    def type_inference(self):
        return self.x.sym_type

    @precondition(allow=VALUE)
    def value_inference(self):
        return np.minimum(np.maximum(self.x.val, self.alpha.val), self.beta.val)


@register_op(doc_str='TODO')
class cos(elementwise_unary):
    def __init__(self, **kwargs):
        super(cos, self).__init__(**kwargs)

    @precondition(allow=VALUE)
    def value_inference(self):
        return np.cos(self.x.val)


@register_op(doc_str='TODO')
class cosh(elementwise_unary):
    def __init__(self, **kwargs):
        super(cosh, self).__init__(**kwargs)

    @precondition(allow=VALUE)
    def value_inference(self):
        return np.cosh(self.x.val)


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

    @precondition(allow=VALUE)
    def value_inference(self):
        return scipy.special.erf(self.x.val)


@register_op(doc_str='TODO')
class exp(elementwise_unary):
    def __init__(self, **kwargs):
        super(exp, self).__init__(**kwargs)

    @precondition(allow=VALUE)
    def value_inference(self):
        return np.exp(self.x.val)


@register_op(doc_str='TODO')
class exp2(elementwise_unary):
    def __init__(self, **kwargs):
        super(exp2, self).__init__(**kwargs)

    @precondition(allow=VALUE)
    def value_inference(self):
        return np.exp2(self.x.val)


@register_op(doc_str='TODO')
class floor(elementwise_unary):
    def __init__(self, **kwargs):
        super(floor, self).__init__(**kwargs)

    @precondition(allow=VALUE)
    def value_inference(self):
        return np.floor(self.x.val)

@register_op(doc_str='TODO')
class inverse(elementwise_unary):
    def __init__(self, **kwargs):
        super(inverse, self).__init__(**kwargs)

    @precondition(allow=VALUE)
    def value_inference(self):
        return np.reciprocal(self.x.val)

@register_op(doc_str='TODO')
class log(elementwise_unary):
    def __init__(self, **kwargs):
        super(log, self).__init__(**kwargs)

    @precondition(allow=VALUE)
    def value_inference(self):
        return np.log(self.x.val)


@register_op(doc_str='TODO')
class logical_not(elementwise_unary):
    def __init__(self, **kwargs):
        super(logical_not, self).__init__(**kwargs)

    def get_operator(self):
        return np.logical_not

    def get_dtype(self, promoted_dtype):
        return types.bool


@register_op(doc_str='TODO')
class round(elementwise_unary):
    def __init__(self, **kwargs):
        super(round, self).__init__(**kwargs)

    @precondition(allow=VALUE)
    def value_inference(self):
        return np.round(self.x.val)


@register_op(doc_str='TODO')
class rsqrt(elementwise_unary):
    def __init__(self, **kwargs):
        super(rsqrt, self).__init__(**kwargs)

    @precondition(allow=VALUE)
    def value_inference(self):
        return 1. / np.sqrt(self.x.val)


@register_op(doc_str='TODO')
class sign(elementwise_unary):
    def __init__(self, **kwargs):
        super(sign, self).__init__(**kwargs)

    @precondition(allow=VALUE)
    def value_inference(self):
        return np.sign(self.x.val)


@register_op(doc_str='TODO')
class sin(elementwise_unary):
    def __init__(self, **kwargs):
        super(sin, self).__init__(**kwargs)

    @precondition(allow=VALUE)
    def value_inference(self):
        return np.sin(self.x.val)


@register_op(doc_str='TODO')
class sinh(elementwise_unary):
    def __init__(self, **kwargs):
        super(sinh, self).__init__(**kwargs)

    @precondition(allow=VALUE)
    def value_inference(self):
        return np.sinh(self.x.val)



@register_op(doc_str='TODO')
class sqrt(elementwise_unary):
    def __init__(self, **kwargs):
        super(sqrt, self).__init__(**kwargs)

    @precondition(allow=VALUE)
    def value_inference(self):
        return np.sqrt(self.x.val)


@register_op(doc_str='TODO')
class square(elementwise_unary):
    def __init__(self, **kwargs):
        super(square, self).__init__(**kwargs)

    @precondition(allow=VALUE)
    def value_inference(self):
        return np.square(self.x.val)


@register_op(doc_str='TODO')
class tan(elementwise_unary):
    def __init__(self, **kwargs):
        super(tan, self).__init__(**kwargs)

    @precondition(allow=VALUE)
    def value_inference(self):
        return np.tan(self.x.val)


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

    @precondition(allow=VALUE)
    def value_inference(self):
        return np.tanh(self.x.val)


@register_op(doc_str='TODO')
class threshold(Operation):
    input_spec = InputSpec(
            x = ScalarOrTensorInputType(),
            alpha = FloatInputType(const=True),
            )

    def __init__(self, **kwargs):
        super(threshold, self).__init__(**kwargs)

    def type_inference(self):
        return self.x.sym_type

    @precondition(allow=VALUE)
    def value_inference(self):
        return np.maximum(self.x.val, self.alpha.val)

@register_op(doc_str='TODO')
class cast(Operation):
    input_spec = InputSpec(
            x = ScalarOrTensorInputType(),
            dtype = StringInputType(const=True)
            )

    def __init__(self, **kwargs):
        super(cast, self).__init__(**kwargs)

    def type_inference(self):
        type_map = {
            "int32": types.int32,
            "int64": types.int64,
            "fp32" : types.fp32,
            "fp64" : types.fp64
        }

        if self.dtype.val not in type_map.keys():
            raise NotImplementedError("Parameter dtype of the cast operation can be one of the {}. "
                                      "Provided {}".format(type_map.keys(), self.dtype.val))

        if not types.is_tensor(self.x.sym_type):
            return type_map[self.dtype.val]

        ret_shape = self.x.shape
        return types.tensor(type_map[self.dtype.val], ret_shape)

    @precondition(allow=VALUE)
    def value_inference(self):
        type_map = {
            "int32": np.int32,
            "int64": np.int64,
            "fp32" : np.float32,
            "fp64" : np.float64
        }

        if self.dtype.val not in type_map.keys():
            raise NotImplementedError("Parameter dtype of the cast operation can be one of the {}. "
                                      "Provided {}".format(type_map.keys(), self.dtype.val))

        if not types.is_tensor(self.x.sym_type):
            return self.x.val.astype(dtype=type_map[self.dtype.val])
        else:
            return np.array(self.x.val).astype(dtype=type_map[self.dtype.val])
