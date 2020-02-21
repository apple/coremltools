from coremltools.converters.nnv2.builtin_types import builtins
from .var import TupleVar
from collections import OrderedDict


class InputSpec(object):
    def __init__(self, **kwargs):
        # Since python 3.6, kwargs preserves the input order. See
        # https://docs.python.org/3/whatsnew/3.6.html#whatsnew36-pep468
        self._input_types = [(k, v) for k, v in kwargs.items()]
        self._ordered_dict = OrderedDict()
        for k, v in self._input_types:
            self._ordered_dict[k] = v

    def update(self, input_types):
        if not isinstance(input_types, InputSpec):
            raise ValueError("Can only add InputSpec to InputTypes")
        self._input_types.extend(input_types._input_types)
        for k, v in input_types._input_types:
            self._ordered_dict[k] = v

    @property
    def input_types(self):
        return self._ordered_dict


class _InputType(object):
    """
    (Untyped) input containing fundamental properties of all inputs to an
    Operation:
    """
    def __init__(self, const=False, default=None, optional=False):
        """
        const (bool):
            True if the InputType has to be constant / materialized at compile time.
            Const InputType is semantically equivalent to attribute. By
            default False. Read-only.

        optional (bool):
            If default is not None, optional will be set to True

        default:
            Default value of optional input. InputType is optional iff a default
            is provided or optional == True.  default can be int, float,
            string, np.ndarray etc depending on subclass.

        Note: _InputType should not be directly instantiated. Only its subclasses may
        be instantiated.
        """
        self.default = default
        self.const = const
        self.optional = True if default is not None else optional

    def is_compatible(self, v):
        """
        v is Var, tuple of Var, or native python function (PyFunctionInput)
        """
        return self._is_compatible(v)

    def _is_compatible(self, v):
        """
        Return True if (possibly symbolic) value `v` is compatible. False
        otherwise.

        Inputs:

        v (builtin value): A possibly symbolic value.

        Comment: Define is_compatible as instance method to call proper subclass
        methods.
        """
        return True

    def _get_predefined_datatype(self):
        """
        Override this function if datatype can be known without `_default` or
        `_val`.
        """
        return None

    @property
    def input_type(self):
        return type(self).__name__


class ScalarOrTensorInputType(_InputType):
    def __init__(self, **kwargs):
        super(ScalarOrTensorInputType, self).__init__(**kwargs)

    def _is_compatible(self, v):
        return builtins.is_scalar(v.dtype) or builtins.is_tensor(v.dtype)


class IntInputType(ScalarOrTensorInputType):
    """
    Int32 input, with _sym_type == builtins.int32

    Set with IntAttribute.val
    Raise error when value set is not integer.
    """
    def __init__(self, **kwargs):
        super(IntInputType, self).__init__(**kwargs)

    def _is_compatible(self, v):
        return v.dtype == builtins.int32

    def _get_predefined_datatype(self):
        return builtins.int32


class BoolInputType(ScalarOrTensorInputType):
    """
    Int32 input, with _sym_type == builtins.int32

    Set with IntAttribute.val
    Raise error when value set is not integer.
    """
    def __init__(self, **kwargs):
        super(BoolInputType, self).__init__(**kwargs)

    def _is_compatible(self, v):
        return v.dtype == builtins.bool

    def _get_predefined_datatype(self):
        return builtins.bool


class FloatInputType(ScalarOrTensorInputType):
    """
    fp32 input, with _sym_type == builtins.fp32

    Set with IntAttribute.val
    Raise error when value set is not integer.
    """
    def __init__(self, **kwargs):
        super(FloatInputType, self).__init__(**kwargs)

    def _is_compatible(self, v):
        return v.dtype == builtins.fp32

    def _get_predefined_datatype(self):
        return builtins.fp32


class TensorInputType(ScalarOrTensorInputType):
    """
    TensorInputType must be numpy ndarray of numeric types. Min rank = 1. (Use
    ScalarOrTensorInputType for possibly scalar input).
    """
    def __init__(self, **kwargs):
        super(TensorInputType, self).__init__(**kwargs)

    def _is_compatible(self, v):
        return builtins.is_tensor(v.sym_type)


class IntTensorInputType(ScalarOrTensorInputType):
    def __init__(self, **kwargs):
        super(IntTensorInputType, self).__init__(**kwargs)

    def _is_compatible(self, v):
        return builtins.is_tensor(v.sym_type) and v.dtype == builtins.int32


class StringInputType(ScalarOrTensorInputType):
    def __init__(self, **kwargs):
        super(StringInputType, self).__init__(**kwargs)

    def _is_compatible(self, v):
        return builtins.is_str(v.sym_type)


class TupleInputType(_InputType):
    def __init__(self, **kwargs):
        super(TupleInputType, self).__init__(**kwargs)

    def _is_compatible(self, v):
        return isinstance(v, TupleVar)


class InternalInputType(_InputType):
    """
    InternalInputType specifies input types outside of Program's type system.
    It allows ops to take, for example, python primitive types, instead of
    only the builtin types.
    """
    def __init__(self, **kwargs):
        super(InternalInputType, self).__init__(**kwargs)

    def _is_compatible(self, v):
        return True  # skip type check by default for InternalInputType.


class PyFunctionInputType(InternalInputType):
    """
    Native python function.
    """
    def __init__(self, **kwargs):
        super(PyFunctionInputType, self).__init__(**kwargs)

    #def _is_compatible(self, v):
    #    return callable(v.val)


class PyTupleInputType(InternalInputType):
    """
    python tuple of Var
    """
    def __init__(self, **kwargs):
        super(PyTupleInputType, self).__init__(**kwargs)

    #def _is_compatible(self, v):
    #    return isinstance(v.val, tuple) and all(isinstance(e, (Var, TupleVar)) for e in v.val)


class InternalStringInputType(InternalInputType):
    def __init__(self, **kwargs):
        super(InternalStringInputType, self).__init__(**kwargs)

    #def _is_compatible(self, v):
    #    return builtins.is_str(v.sym_type)


class InternalScalarOrTensorInputType(InternalInputType):
    def __init__(self, **kwargs):
        super(InternalScalarOrTensorInputType, self).__init__(**kwargs)

    #def _is_compatible(self, v):
    #    return builtins.is_scalar(v.dtype) or builtins.is_tensor(v.dtype)
