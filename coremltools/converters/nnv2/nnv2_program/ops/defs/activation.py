import scipy
from ._op_reqs import *
from .elementwise_unary import elementwise_unary

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
    input_spec = InputSpec(
            x = ScalarOrTensorInputType(),
            alpha = FloatInputType(const=True),
            beta = FloatInputType(const=True),
            )

    def __init__(self, **kwargs):
        super(clamped_relu, self).__init__(**kwargs)

    @precondition(allow=VALUE)
    def value_inference(self):
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
    input_spec = InputSpec(
            x = ScalarOrTensorInputType(),
            alpha = FloatInputType(const=True, default=1),
            )

    def __init__(self, **kwargs):
        super(elu, self).__init__(**kwargs)

    @precondition(allow=VALUE)
    def value_inference(self):
        b = np.copy(self.x.val)
        b[b < 0] = self.alpha.val * (np.exp(b[b < 0]) - 1)
        return b

    def type_inference(self):
        return self.x.sym_type


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

    @precondition(allow=VALUE)
    def value_inference(self):
        return 0.5 * self.x.val * (1 + scipy.special.erf(self.x.val / np.sqrt(2)))


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
    input_spec = InputSpec(
            x = ScalarOrTensorInputType(),
            alpha = FloatInputType(const=True, default=0.01),
        )

    def __init__(self, **kwargs):
        super(leaky_relu, self).__init__(**kwargs)

    @precondition(allow=VALUE)
    def value_inference(self):
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
    input_spec = InputSpec(
            x = ScalarOrTensorInputType(),
            alpha = FloatInputType(const=True),
            beta = FloatInputType(const=True, default=0.),
            )

    def __init__(self, **kwargs):
        super(linear_activation, self).__init__(**kwargs)

    @precondition(allow=VALUE)
    def value_inference(self):
        return self.alpha.val * self.x.val + self.beta.val

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
    input_spec = InputSpec(
            x = TensorInputType(),
            alpha = TensorInputType(const=True),
            )

    def __init__(self, **kwargs):
        super(prelu, self).__init__(**kwargs)

    @precondition(allow=VALUE)
    def value_inference(self):
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

    @precondition(allow=VALUE)
    def value_inference(self):
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

    @precondition(allow=VALUE)
    def value_inference(self):
        return np.minimum(np.maximum(self.x.val, 0), 6)


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
    input_spec = InputSpec(
            x = ScalarOrTensorInputType(),
            alpha = FloatInputType(const=True, default=1),
            beta = FloatInputType(const=True, default=1),
            )

    def __init__(self, **kwargs):
        super(scaled_tanh, self).__init__(**kwargs)

    @precondition(allow=VALUE)
    def value_inference(self):
        return self.alpha.val * np.tanh(self.x.val * self.beta.val)

    def type_inference(self):
        return self.x.sym_type


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

    @precondition(allow=VALUE)
    def value_inference(self):
        return 1/(1 + np.exp(-self.x.val))


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
    input_spec = InputSpec(
            x = ScalarOrTensorInputType(),
            alpha = FloatInputType(const=True, default=0.2),
            beta = FloatInputType(const=True, default=0.5),
            )

    def __init__(self, **kwargs):
        super(sigmoid_hard, self).__init__(**kwargs)

    @precondition(allow=VALUE)
    def value_inference(self):
        return np.minimum(np.maximum((self.alpha.val * self.x.val) + self.beta.val, 0), 1)

    def type_inference(self):
        return self.x.sym_type


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

     @precondition(allow=VALUE)
     def value_inference(self):
         return np.log(1 + np.exp(-np.abs(self.x.val))) + np.maximum(self.x.val, 0)


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
    input_spec = InputSpec(
            x = TensorInputType(),
            alpha = TensorInputType(const=True),
            beta = TensorInputType(const=True),
            )

    def __init__(self, **kwargs):
        super(softplus_parametric, self).__init__(**kwargs)

    @precondition(allow=VALUE)
    def value_inference(self):
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


@register_op(doc_str='''
Returns exp(x) / tf.reduce_sum(tf.exp(x), axis)

Parameters
----------
x: <*, f32>, required
axis: const<f32>, optional, default = -1

Returns
-------
<*, f32>, a tensor of the same shape as x.

''')
class softmax(Operation):
    input_spec = InputSpec(
            logit = TensorInputType(),
            axis = IntInputType(const=True, default=-1),
            )

    def __init__(self, **kwargs):
        super(softmax, self).__init__(**kwargs)

    def type_inference(self):
        return self.logit.sym_type

    @precondition(allow=VALUE)
    def value_inference(self):
        x = self.logit.val
        axis = self.axis.val
        return scipy.special.softmax(x, axis=axis)

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

    @precondition(allow=VALUE)
    def value_inference(self):
        return self.x.val / (1 + np.abs(self.x.val))


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
    input_spec = InputSpec(
            x = ScalarOrTensorInputType(),
            alpha = FloatInputType(const=True, default=1),
            )

    def __init__(self, **kwargs):
        super(thresholded_relu, self).__init__(**kwargs)

    def type_inference(self):
        return self.x.sym_type

    @precondition(allow=VALUE)
    def value_inference(self):
        return np.maximum(self.x.val - self.alpha.val, 0)
