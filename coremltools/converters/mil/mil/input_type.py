#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from coremltools.converters.mil.mil import types
from .var import InternalVar
from collections import OrderedDict

SUPPORT_INT_TYPES = [
                        types.uint8,
                        types.int8,
                        types.uint16,
                        types.int16,
                        types.uint32,
                        types.int32,
                        types.uint64,
                        types.int64,
                    ]

SUPPORT_FLOAT_TYPES = [
                        types.fp16,
                        types.fp32,
                        types.fp64,
                    ]

class InputTypeError(Exception):
    def __init__(self, message, input_arg_name=None, input_param_name=None,
        input_type=None,
            actual_type=None):
        super(InputTypeError, self).__init__(message)
        self.input_arg_name = input_arg_name
        self.input_param_name = input_param_name
        self.input_type = input_type
        self.actual_type = actual_type

class InputSpec(object):
    def __init__(self, **kwargs):
        # Since python 3.6, kwargs preserves the input order. See
        # https://docs.python.org/3/whatsnew/3.6.html#whatsnew36-pep468
        self._input_types = [(k, v) for k, v in kwargs.items()]
        self._ordered_dict = OrderedDict()
        for k, v in self._input_types:
            self._ordered_dict[k] = v

    def __add__(self, input_spec):
        self._input_types.extend(input_spec._input_types)
        for k, v in input_spec._input_types:
            self._ordered_dict[k] = v
        return self

    @property
    def input_types(self):
        """
        Ordered dict[str, _InputType] (name, input_type)
        """
        return self._ordered_dict

    def parse_inputs(self, kwargs):
        """ Parse and extract (name, value) pairs from kwargs according to the spec.

        Args:
            kwargs: must contain a Var compatible with
                    compatible type for each
                    1) required _InputType
                    2) optional _InputType with default value

        Return:
            out: List[(name, Var or None)]
                The list has the same length as the `input_types`.
                `(k, None)` is in the list iff input_type of `k`
                is optional, has no default value, and
                `k` is not specified in the input.

        Raise:
            TypeError if value type is incompatible
            ValueError if a require input is missing
        """
        ret = []
        no_check_var_visibility = kwargs.get("no_check_var_visibility", False)
        for name, input_type in self.input_types.items():
            if name in kwargs:
                var = kwargs[name]
                # TODO (jay): we should remove this internal var later as we
                # further cleanup the interface
                if isinstance(var, InternalVar) or input_type.is_compatible(var):
                    ret.append((name, var))
                else:
                    msg = (
                        "Input {} has type {} not compatible with "
                        "expected type {}".format(name, var.sym_type, input_type)
                    )
                    raise InputTypeError(msg, name, var.name,
                            input_type=input_type,
                            actual_type=var.sym_type)
            else:
                # if input is not found in kwargs, it must be optional has no
                # default value
                if not input_type.optional or input_type.default:
                    # Skip check on PyFunctionInput since created while_loop /
                    # cond ops don't need to rebuild the nested blocks
                    if no_check_var_visibility or isinstance(
                        input_type, PyFunctionInputType
                    ):
                        continue
                    raise ValueError("Input {} is required".format(name))
                else:
                    assert input_type.default is None
                    ret.append((name, None))
        return ret


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
            Default value of optional input. InputType is optional if a default
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
        Return True if (possibly symbolic) value `v` is compatible. False
        otherwise.

        Inputs:

        v (Var | ListVar | native python function): input

        Comment: Define is_compatible as instance method to call proper subclass
        methods.
        """
        return self._is_compatible(v)

    def _is_compatible(self, v):
        return True

    def _get_predefined_datatype(self):
        """
        Override this function if datatype can be known without `_default` or
        `_val`.
        """
        return None

    def __str__(self):
        return type(self).__name__

    @property
    def type_str(self):
        """Descriptive string describing expected mil types"""
        return self.__str__(self)

class ListInputType(_InputType):
    def __init__(self, **kwargs):
        super(ListInputType, self).__init__(**kwargs)

    def _is_compatible(self, v):
        return types.is_list(v.sym_type)

    @property
    def type_str(self):
        return 'list'

class ScalarOrTensorInputType(_InputType):
    def __init__(self, **kwargs):
        super(ScalarOrTensorInputType, self).__init__(**kwargs)

    def _is_compatible(self, v):
        return types.is_scalar(v.dtype) or types.is_tensor(v.dtype)

    @property
    def type_str(self):
        return 'tensor or scalar'


class ListOrScalarOrTensorInputType(_InputType):
    def __init__(self, **kwargs):
        super(ListOrScalarOrTensorInputType, self).__init__(**kwargs)

    def _is_compatible(self, v):
        return (
            types.is_list(v.sym_type)
            or types.is_scalar(v.dtype)
            or types.is_tensor(v.dtype)
        )

    @property
    def type_str(self):
        return 'list, tensor, or scalar'


class IntInputType(ScalarOrTensorInputType):
    """
    Int input with _sym_type in [types.uint8, types.int8, types.uint16, types.int16,
                                 types.uint32, types.int32, types.uint64, types.int64]
    predefined to be types.int32 by default.

    Set with IntAttribute.val
    Raise error when value set is not integer.
    """

    def __init__(self, **kwargs):
        super(IntInputType, self).__init__(**kwargs)

    def _is_compatible(self, v):
        return v.dtype in SUPPORT_INT_TYPES

    def _get_predefined_datatype(self):
        return types.int32

    @property
    def type_str(self):
        return 'integer tensor or scalar'

class BoolInputType(ScalarOrTensorInputType):
    """
    Int32 input, with _sym_type == types.int32

    Set with IntAttribute.val
    Raise error when value set is not integer.
    """

    def __init__(self, **kwargs):
        super(BoolInputType, self).__init__(**kwargs)

    def _is_compatible(self, v):
        return v.dtype == types.bool

    def _get_predefined_datatype(self):
        return types.bool

    @property
    def type_str(self):
        return 'bool tensor or scalar'

class FloatInputType(ScalarOrTensorInputType):
    """
    fp32 input, with _sym_type == types.fp32

    Set with IntAttribute.val
    Raise error when value set is not integer.
    """

    def __init__(self, **kwargs):
        super(FloatInputType, self).__init__(**kwargs)

    def _is_compatible(self, v):
        return v.dtype in SUPPORT_FLOAT_TYPES

    def _get_predefined_datatype(self):
        return types.fp32

    @property
    def type_str(self):
        return 'float tensor or scalar'

class IntOrFloatInputType(ScalarOrTensorInputType):
    """
    input with _sym_type in [types.uint8, types.int8, types.uint16, types.int16,
                             types.uint32, types.int32, types.uint64, types.int64,
                             types.fp32]
    predefined to be types.fp32 by default.
    """

    def __init__(self, **kwargs):
        super(IntOrFloatInputType, self).__init__(**kwargs)

    def _is_compatible(self, v):
        return v.dtype in SUPPORT_INT_TYPES + SUPPORT_FLOAT_TYPES


    def _get_predefined_datatype(self):
        return types.fp32

    @property
    def type_str(self):
        return 'integer, float tensor or scalar'

class IntOrFloatOrBoolInputType(ScalarOrTensorInputType):
    """
    input with _sym_type in [types.uint8, types.int8, types.uint16, types.int16,
                             types.uint32, types.int32, types.uint64, types.int64,
                             types.fp32, types.bool]
    predefined to be types.fp32 by default.
    """

    def __init__(self, **kwargs):
        super(IntOrFloatOrBoolInputType, self).__init__(**kwargs)

    def _is_compatible(self, v):
        return v.dtype in SUPPORT_INT_TYPES + SUPPORT_FLOAT_TYPES + [types.bool]

    def _get_predefined_datatype(self):
        return types.fp32

    @property
    def type_str(self):
        return 'integer, float, bool tensor or scalar'

class TensorInputType(ScalarOrTensorInputType):
    """
    TensorInputType must be numpy ndarray of numeric types. Min rank = 1. (Use
    ScalarOrTensorInputType for possibly scalar input).
    """

    def __init__(self, **kwargs):
        super(TensorInputType, self).__init__(**kwargs)

    def _is_compatible(self, v):
        # We only support scalar string type.
        return types.is_tensor(v.sym_type) and \
            v.sym_type.get_primitive() != types.str

    @property
    def type_str(self):
        return 'tensor'

class IntTensorInputType(ScalarOrTensorInputType):
    """
    Tensor input with int values
    with _sym_type in [types.uint8, types.int8, types.uint16, types.int16,
                       types.uint32, types.int32, types.uint64, types.int64]

    Raise error when value set is not integer.
    """

    def __init__(self, **kwargs):
        super(IntTensorInputType, self).__init__(**kwargs)

    def _is_compatible(self, v):
        return types.is_tensor(v.sym_type) and v.dtype in SUPPORT_INT_TYPES
    @property
    def type_str(self):
        return 'integer tensor'

class BoolTensorInputType(ScalarOrTensorInputType):
    def __init__(self, **kwargs):
        super(BoolTensorInputType, self).__init__(**kwargs)

    def _is_compatible(self, v):
        return types.is_tensor(v.sym_type) and v.dtype == types.bool

    @property
    def type_str(self):
        return 'bool tensor'

class StringInputType(ScalarOrTensorInputType):
    def __init__(self, **kwargs):
        super(StringInputType, self).__init__(**kwargs)

    def _is_compatible(self, v):
        return types.is_str(v.sym_type)

    @property
    def type_str(self):
        return 'str'

class TupleInputType(_InputType):
    def __init__(self, **kwargs):
        super(TupleInputType, self).__init__(**kwargs)

    def _is_compatible(self, v):
        # We don't check the detail types within the tuple.
        return isinstance(v, (tuple, list))

    @property
    def type_str(self):
        return 'tuple'

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

    # def _is_compatible(self, v):
    #    return callable(v.val)


class InternalStringInputType(InternalInputType):
    def __init__(self, **kwargs):
        super(InternalStringInputType, self).__init__(**kwargs)

    # def _is_compatible(self, v):
    #    return types.is_str(v.sym_type)


class InternalScalarOrTensorInputType(InternalInputType):
    def __init__(self, **kwargs):
        super(InternalScalarOrTensorInputType, self).__init__(**kwargs)

    # def _is_compatible(self, v):
    #    return types.is_scalar(v.dtype) or types.is_tensor(v.dtype)
