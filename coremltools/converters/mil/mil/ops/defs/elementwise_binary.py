#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause
import numpy as np

import operator
from ._op_reqs import *
from ._utils import promoted_primitive_type, broadcast_shapes


class elementwise_binary(Operation):
    """
    Elementwise Binary Op Superclass
    """
    input_spec = InputSpec(x=ScalarOrTensorInputType(), y=ScalarOrTensorInputType(),)

    def __init__(self, **kwargs):
        super(elementwise_binary, self).__init__(**kwargs)

    def type_inference(self):
        typea = self.x.sym_type
        typeb = self.y.sym_type
        primitive_type = promoted_primitive_type(typea, typeb)
        if primitive_type is None:
            raise ValueError("Incompatible primitive types in broadcast operation")
        primitive_type = self.get_dtype(primitive_type)

        # broadcast
        if not types.is_tensor(typea) and not types.is_tensor(typeb):
            # both typea and typeb are not tensors
            return primitive_type
        if types.is_tensor(typea) and not types.is_tensor(typeb):
            # a is tensor, b is not
            return types.tensor(primitive_type, typea.get_shape())
        if not types.is_tensor(typea) and types.is_tensor(typeb):
            # a is not tensor, b is
            return types.tensor(primitive_type, typeb.get_shape())

        # both a, b are tensors
        shapea = list(typea.get_shape())
        shapeb = list(typeb.get_shape())
        ret_shape = broadcast_shapes(shapea, shapeb)
        return types.tensor(primitive_type, ret_shape)

    @precondition(allow=VALUE)
    def value_inference(self):
        return self._cast_check_value_inferene(self.x.val, self.y.val)

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

    def _cast_check_value_inferene(self, a, b):
        """
        If one of the input is tensor, cast the result to tensor.
        """
        to_cast = any([isinstance(x, np.ndarray) for x in [a, b]])
        result = self.get_operator()(a, b)
        return result if not to_cast else np.array(result)


"""
Elementwise Binary Op Implementation(s)
"""


@register_op(doc_str="")
class add(elementwise_binary):
    """
    Return ``x + y`` element-wise with
    `broadcasting <https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html>`_.
    
    Parameters
    ----------
    x: <\*,T> (Required)
        * Shape must be compatible with ``y`` in broadcast.
    
    y: <\*,T> (Required)
        * Shape must be compatible with ``x`` in broadcast.
    
    Returns
    -------
    <\*,T>
    
    Attributes
    ----------
    T: fp32
    """
    
    def __init__(self, **kwargs):
        super(add, self).__init__(**kwargs)

    def get_operator(self):
        return operator.add


@register_op(doc_str="")
class equal(elementwise_binary):
    """
    Return the truth value of ``x == y`` element-wise with
    `broadcasting <https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html>`_
    (``1`` for true, ``0`` for false in numeric domain).
    
    Parameters
    ----------
    x: <\*,T> (Required)
        * Shape must be compatible with ``y`` in broadcast.
    
    y: <\*,T> (Required)
        * Shape must be compatible with ``x`` in broadcast.
    
    Returns
    -------
    <\*, bool>
        * A boolean tensor with the same shape as the inputs.
    
    Attributes
    ----------
    T: fp32
    """

    def __init__(self, **kwargs):
        super(equal, self).__init__(**kwargs)

    def get_operator(self):
        return np.equal

    def get_dtype(self, promoted_dtype):
        return types.bool


@register_op(doc_str="")
class floor_div(elementwise_binary):
    """
    Return ``x / y`` element-wise with
    `broadcasting <https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html>`_,
    rounded towards negative infinity.
    
    Parameters
    ----------
    x: tensor<\*, T> (Required)
        * Shape must be compatible with ``y`` in broadcast.
    
    y: tensor<\*, T> (Required)
        * Shape must be compatible with ``x`` in broadcast.
    
    Returns
    -------
    tensor<\*, T>
        * A tensor of the same type and shape as the inputs.
    
    Attributes
    ----------
    T: fp32
    """
    
    def __init__(self, **kwargs):
        super(floor_div, self).__init__(**kwargs)

    def get_operator(self):
        return operator.floordiv


@register_op(doc_str="")
class greater(elementwise_binary):
    """
    Return the truth value of ``x > y`` element-wise with
    `broadcasting <https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html>`_
    (``1`` for true, ``0`` for false in numeric domain).

    Parameters
    ----------
    x: tensor<\*, T> (Required)
        * Shape must be compatible with ``y`` in broadcast.
    
    y: tensor<\*, T> (Required)
        * Shape must be compatible with ``x`` in broadcast.
    
    Returns
    -------
    tensor<\*, bool>
        * A boolean tensor with the same shape as the inputs.

    Attributes
    ----------
    T: fp32
    """

    def __init__(self, **kwargs):
        super(greater, self).__init__(**kwargs)

    def get_operator(self):
        return operator.gt

    def get_dtype(self, promoted_dtype):
        return types.bool


@register_op(doc_str="")
class greater_equal(elementwise_binary):
    """
    Return the truth value of ``x >= y`` element-wise with
    `broadcasting <https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html>`_
    (``1`` for true, ``0`` for false in numeric domain).
    
    Parameters
    ----------
    x: tensor<\*, T> (Required)
        * Shape must be compatible with ``y`` in broadcast.
    
    y: tensor<\*, T> (Required)
        * Shape must be compatible with ``x`` in broadcast.
    
    Returns
    -------
    tensor<\*?, bool>
        * A boolean tensor with the same shape as the inputs.
    
    Attributes
    ----------
    T: fp32
    """

    def __init__(self, **kwargs):
        super(greater_equal, self).__init__(**kwargs)

    def get_operator(self):
        return operator.ge

    def get_dtype(self, promoted_dtype):
        return types.bool


@register_op(doc_str="")
class less(elementwise_binary):
    """
    Return the truth value of ``x < y`` element-wise with
    `broadcasting <https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html>`_
    (``1`` for true, ``0`` for false in numeric domain).
    
    Parameters
    ----------
    x: tensor<\*, T> (Required)
        * Shape must be compatible with ``y`` in broadcast.
    
    y: tensor<\*, T> (Required)
        * Shape must be compatible with ``x`` in broadcast.
    
    Returns
    -------
    tensor<\*?, bool>
        * A boolean tensor with the same shape as the inputs.
    
    Attributes
    ----------
    T: fp32
    """

    def __init__(self, **kwargs):
        super(less, self).__init__(**kwargs)

    def get_operator(self):
        return operator.lt

    def get_dtype(self, promoted_dtype):
        return types.bool


@register_op(doc_str="")
class less_equal(elementwise_binary):
    """
    Return the truth value of ``x <= y`` element-wise with
    `broadcasting <https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html>`_
    (``1`` for true, ``0`` for false in numeric domain).
    
    Parameters
    ----------
    x: tensor<\*, T> (Required)
        * Shape must be compatible with ``y`` in broadcast.
    
    y: tensor<\*, T> (Required)
        * Shape must be compatible with ``x`` in broadcast.
    
    Returns
    -------
    tensor<\*?, bool>
        * A boolean tensor with the same shape as the inputs.
    
    Attributes
    ----------
    T: fp32
    """
    
    def __init__(self, **kwargs):
        super(less_equal, self).__init__(**kwargs)

    def get_operator(self):
        return operator.le

    def get_dtype(self, promoted_dtype):
        return types.bool


@register_op(doc_str="")
class logical_and(elementwise_binary):
    """
    Return the truth value of ``x AND y`` element-wise with
    `broadcasting <https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html>`_
    (``1`` for true, ``0`` for false in numeric domain). A numeric value ``t`` is
    evaluated to true if ``t != 0``.

    Parameters
    ----------
    x: tensor<\*, T> (Required)
        * Shape must be compatible with ``y`` in broadcast.
    
    y: tensor<\*, T> (Required)
        * Shape must be compatible with ``x`` in broadcast.
    
    Returns
    -------
    tensor<\*?, bool>
        * A boolean tensor with the same shape as the inputs.
    
    Attributes
    ----------
    T: fp32
    
    """
    
    def __init__(self, **kwargs):
        super(logical_and, self).__init__(**kwargs)

    def get_operator(self):
        return np.logical_and

    def get_dtype(self, promoted_dtype):
        return types.bool


@register_op(doc_str="")
class logical_or(elementwise_binary):
    """
    Return the truth value of ``x OR y`` element-wise with
    `broadcasting <https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html>`_
    (``1`` for true, ``0`` for false in numeric domain). A numeric value ``t`` is
    evaluated to true if ``t != 0``.
    
    Parameters
    ----------
    x: tensor<\*, T> (Required)
        * Shape must be compatible with ``y`` in broadcast.
    
    y: tensor<\*, T> (Required)
        * Shape must be compatible with ``x`` in broadcast.
    
    Returns
    -------
    tensor<\*?, bool>
        * A boolean tensor with the same shape as the inputs.
    
    Attributes
    ----------
    T: fp32
    
    """
    
    def __init__(self, **kwargs):
        super(logical_or, self).__init__(**kwargs)

    def get_operator(self):
        return np.logical_or

    def get_dtype(self, promoted_dtype):
        return types.bool


@register_op(doc_str="")
class logical_xor(elementwise_binary):
    """
    Return the truth value of ``x XOR y`` element-wise with
    `broadcasting <https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html>`_
    (``1`` for true, ``0`` for false in numeric domain). A numeric value ``t`` is
    evaluated to true if ``t != 0``.
    
    Parameters
    ----------
    x: tensor<\*, T> (Required)
        * Shape must be compatible with ``y`` in broadcast.
    
    y: tensor<\*, T> (Required)
        * Shape must be compatible with ``x`` in broadcast.
    
    Returns
    -------
    tensor<\*?, bool>
        * A boolean tensor with the same shape as the inputs.
    
    Attributes
    ----------
    T: fp32
    
    """
    
    def __init__(self, **kwargs):
        super(logical_xor, self).__init__(**kwargs)

    def get_operator(self):
        return np.logical_xor

    def get_dtype(self, promoted_dtype):
        return types.bool


@register_op(doc_str="")
class maximum(elementwise_binary):
    """
    Return ``x > y ? x : y`` element-wise with
    `broadcasting <https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html>`_.
    
    Parameters
    ----------
    x: tensor<\*, T> (Required)
        * Shape must be compatible with ``y`` in broadcast.
    
    y: tensor<\*, T> (Required)
        * Shape must be compatible with ``x`` in broadcast.
    
    Returns
    -------
    tensor<\*?, bool>
        * A boolean tensor with the same shape as the inputs.
    
    Attributes
    ----------
    T: fp32
    """
    
    def __init__(self, **kwargs):
        super(maximum, self).__init__(**kwargs)

    def get_operator(self):
        return np.maximum


@register_op(doc_str="")
class minimum(elementwise_binary):
    """
    Return ``x > y ? y : x`` element-wise with
    `broadcasting <https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html>`_.
    
    Parameters
    ----------
    x: tensor<\*, T> (Required)
        * Shape must be compatible with ``y`` in broadcast.
    
    y: tensor<\*, T> (Required)
        * Shape must be compatible with ``x`` in broadcast.
    
    Returns
    -------
    tensor<\*?, bool>
        * A boolean tensor with the same shape as the inputs.
    
    Attributes
    ----------
    T: fp32
    """
    
    def __init__(self, **kwargs):
        super(minimum, self).__init__(**kwargs)

    def get_operator(self):
        return np.minimum


@register_op(doc_str="")
class mod(elementwise_binary):
    """
    Return ``x % y`` element-wise with
    `broadcasting <https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html>`_.

    Parameters
    ----------
    x: tensor<\*, T> (Required)
        * Shape must be compatible with ``y`` in broadcast.
    
    y: tensor<\*, T> (Required)
        * Shape must be compatible with ``x`` in broadcast.
    
    Returns
    -------
    tensor<\*?, bool>
        * A boolean tensor with the same shape as the inputs.
    
    Attributes
    ----------
    T: fp32
    """
    
    def __init__(self, **kwargs):
        super(mod, self).__init__(**kwargs)

    def get_operator(self):
        return operator.mod


@register_op(doc_str="")
class mul(elementwise_binary):
    """
    Return ``x * y`` element-wise with
    `broadcasting <https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html>`_.

    Parameters
    ----------
    x: tensor<\*, T> (Required)
        * Shape must be compatible with ``y`` in broadcast.
    
    y: tensor<\*, T> (Required)
        * Shape must be compatible with ``x`` in broadcast.
    
    Returns
    -------
    tensor<\*?, bool>
        * A boolean tensor with the same shape as the inputs.
    
    Attributes
    ----------
    T: fp32
    """
    
    def __init__(self, **kwargs):
        super(mul, self).__init__(**kwargs)

    def get_operator(self):
        return operator.mul


@register_op(doc_str="")
class not_equal(elementwise_binary):
    """
    Return the truth value of ``x != y`` element-wise with
    `broadcasting <https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html>`_
    (``1`` for true, ``0`` for false in numeric domain).
    
    Parameters
    ----------
    x: tensor<\*, T> (Required)
        * Shape must be compatible with ``y`` in broadcast.
    
    y: tensor<\*, T> (Required)
        * Shape must be compatible with ``x`` in broadcast.
    
    Returns
    -------
    tensor<\*?, bool>
        * A boolean tensor with the same shape as the inputs.

    Attributes
    ----------
    T: fp32
    """
    
    def __init__(self, **kwargs):
        super(not_equal, self).__init__(**kwargs)

    def get_operator(self):
        return operator.ne

    def get_dtype(self, promoted_dtype):
        return types.bool


@register_op(doc_str="")
class real_div(elementwise_binary):
    """
    Return ``x / y`` element-wise with
    `broadcasting <https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html>`_.

    Parameters
    ----------
    x: tensor<\*, T> (Required)
        * Shape must be compatible with ``y`` in broadcast.
    
    y: tensor<\*, T> (Required)
        * Shape must be compatible with ``x`` in broadcast.
    
    Returns
    -------
    tensor<\*?, T>
        * A tensor with the same type and shape as the inputs.
    
    Attributes
    ----------
    T: fp32
    """
    
    def __init__(self, **kwargs):
        # TODO(rdar://79925291): Allow int32 input to floor_div
        from coremltools.converters.mil.mil import Builder as mb
        from coremltools.converters.mil.mil import types
        accepted_types = [types.fp32, types.fp16]
        for input_name in ["x", "y"]:
            if kwargs[input_name].dtype not in accepted_types:
                kwargs[input_name] = mb.cast(x=kwargs[input_name], dtype="fp32")
        super(real_div, self).__init__(**kwargs)

    def get_operator(self):
        return operator.truediv


@register_op(doc_str="")
class pow(elementwise_binary):
    """
    Return ``x ^ y`` element-wise with
    `broadcasting <https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html>`_.

    Parameters
    ----------
    x: tensor<\*, T> (Required)
        * Shape must be compatible with ``y`` in broadcast.
    
    y: tensor<\*, T> (Required)
        * Shape must be compatible with ``x`` in broadcast.
    
    Returns
    -------
    tensor<\*?, bool>
        * A boolean tensor with the same shape as the inputs.
    
    Attributes
    ----------
    T: fp32
    """
    
    def __init__(self, **kwargs):
        super(pow, self).__init__(**kwargs)

    def get_operator(self):
        return operator.pow


@register_op(doc_str="")
class sub(elementwise_binary):
    """
    Return ``x - y`` element-wise with
    `broadcasting <https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html>`_.

    Parameters
    ----------
    x: tensor<\*, T> (Required)
        * Shape must be compatible with ``y`` in broadcast.
    
    y: tensor<\*, T> (Required)
        * Shape must be compatible with ``x`` in broadcast.
    
    Returns
    -------
    tensor<\*?, bool>
        * A boolean tensor with the same shape as the inputs.
    
    Attributes
    ----------
    T: fp32
    """
    
    def __init__(self, **kwargs):
        super(sub, self).__init__(**kwargs)

    def get_operator(self):
        return operator.sub
