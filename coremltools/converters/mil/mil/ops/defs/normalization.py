#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from ._op_reqs import *


@register_op(doc_str="TODO")
class batch_norm(Operation):
    input_spec = InputSpec(
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


@register_op(doc_str="TODO")
class instance_norm(Operation):
    input_spec = InputSpec(
        x=TensorInputType(),
        gamma=TensorInputType(const=True, optional=True),
        beta=TensorInputType(const=True, optional=True),
        epsilon=FloatInputType(const=True, default=1e-5),
    )

    def __init__(self, **kwargs):
        super(instance_norm, self).__init__(**kwargs)

    def type_inference(self):
        return self.x.sym_type


@register_op(doc_str="TODO")
class l2_norm(Operation):
    input_spec = InputSpec(
        x=TensorInputType(),
        axes=IntTensorInputType(),
        epsilon=FloatInputType(const=True, default=1e-12),
    )

    def __init__(self, **kwargs):
        super(l2_norm, self).__init__(**kwargs)

    def type_inference(self):
        return self.x.sym_type


@register_op(doc_str="TODO")
class layer_norm(Operation):
    input_spec = InputSpec(
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

    @precondition(allow=VALUE)
    def value_inference(self):
        def np_layer_norm(x, axes, gamma, beta, epsilon=1e-5):
            normalized_shape = x.shape[-len(axes) :]
            gamma = np.ones(shape=normalized_shape) if gamma is None else gamma
            beta = np.zeros(shape=normalized_shape) if beta is None else beta
            num = x - np.mean(x, axis=tuple(axes), keepdims=True)
            dem = np.sqrt(
                np.sum(np.square(num), axis=tuple(axes), keepdims=True)
                / np.prod(normalized_shape)
                + epsilon
            )
            return num / dem * gamma + beta

        _axes = self.x.shape if self.axes is None else self.axes.val
        _gamma = None if self.gamma is None else self.gamma.val
        _beta = None if self.beta is None else self.beta.val
        return np_layer_norm(self.x.val, _axes, _gamma, _beta, self.epsilon.val)


@register_op(doc_str="TODO")
class local_response_norm(Operation):
    input_spec = InputSpec(
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
