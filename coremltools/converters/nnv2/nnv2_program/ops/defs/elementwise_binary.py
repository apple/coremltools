import operator
from ._op_reqs import *
from ._utils import promoted_primitive_type, broadcast_shapes

"""
Elementwise Binary Op Superclass
"""
class elementwise_binary(Operation):
    input_spec = InputSpec(
            x = ScalarOrTensorInputType(),
            y = ScalarOrTensorInputType(),
            )

    def __init__(self, **kwargs):
        super(elementwise_binary, self).__init__(**kwargs)

    def type_inference(self):
        typea = self.x.sym_type
        typeb = self.y.sym_type
        primitive_type = promoted_primitive_type(typea, typeb)
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
        ret_shape = broadcast_shapes(shapea, shapeb)
        return builtins.tensor(primitive_type, ret_shape)

    @precondition(allow=VALUE)
    def value_inference(self):
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


"""
Elementwise Binary Op Implmentation(s)
"""

@register_op(doc_str='TODO')
class add(elementwise_binary):
    def __init__(self, **kwargs):
        super(add, self).__init__(**kwargs)

    def get_operator(self):
        return operator.add


@register_op(doc_str='TODO')
class equal(elementwise_binary):
    def __init__(self, **kwargs):
        super(equal, self).__init__(**kwargs)

    def get_operator(self):
        return np.equal

    def get_dtype(self, promoted_dtype):
        return builtins.bool


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
class logical_and(elementwise_binary):
    def __init__(self, **kwargs):
        super(logical_and, self).__init__(**kwargs)

    def get_operator(self):
        return np.logical_and

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
class pow(elementwise_binary):
    def __init__(self, **kwargs):
        super(pow, self).__init__(**kwargs)

    def get_operator(self):
        return operator.pow


@register_op(doc_str='TODO')
class sub(elementwise_binary):
    def __init__(self, **kwargs):
        super(sub, self).__init__(**kwargs)

    def get_operator(self):
        return operator.sub
