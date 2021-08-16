#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from collections import OrderedDict

from coremltools.converters.mil.mil import types
from coremltools.converters.mil.mil.var import InternalVar


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

class DefaultInputs(object):
    def __init__(self, **kwargs):
        # Since python 3.6, kwargs preserves the input order. See
        # https://docs.python.org/3/whatsnew/3.6.html#whatsnew36-pep468
        self._default_inputs = [(k, v) for k, v in kwargs.items()]
        self._ordered_dict = OrderedDict()
        for k, v in self._default_inputs:
            self._ordered_dict[k] = v

    def items(self):
        return self._ordered_dict.items()

    def __add__(self, default_inputs):
        self._default_inputs.extend(default_inputs._default_inputs)
        for k, v in default_inputs._default_inputs:
            self._ordered_dict[k] = v
        return self

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

    def validate_inputs(self, op_name, op_type, candidate_kvs):
        """
        For each key K in `candidate_kvs`, if K is found in
        self.input_types, perform the followings:

        - check that candidate_kvs[K] is a Var and satisfies
        requirements in InputType (const, types)
        - Place K, candidate_kvs[K] in output (list of (name, var) pairs).

        Note that this does not ensure the presence of all required
        input_spec (optional == False).

        Parameters
        ----------
        - op_name: str

        - op_type: str

        - candidate_kvs: Dict[str, Var]
          Values cannot be None

        Return
        ------
        None

        Raise:
            ValueErrr if value type is incompatible
        """
        msg_prefix = 'Op \"{}\" (op_type: {}) '.format(op_name, op_type)

        # Ensure candidate_kvs doesn't contain None
        for name, var in candidate_kvs.items():
            if var is None:
                raise ValueError(msg_prefix + 'Input {} is None'.format(name))

            if name not in self.input_types:
                raise ValueError(msg_prefix + \
                    'Unrecognized input {}'.format(name))

            input_type = self.input_types[name]
            # Check constness
            # Don't check InternalInputType (so _const_symbolic can work)
            if input_type.const and \
                not isinstance(input_type, InternalInputType) \
                and var.val is None:
                msg = msg_prefix + \
                    'Input {} must be const at compile time'
                raise ValueError(msg.format(name), name, var.name)

            if not isinstance(var, InternalVar) and \
                not input_type.is_compatible(var):
                msg = msg_prefix + "Input {}=\"{}\" expects " +\
                        "{} but got {}"
                raise ValueError(msg.format(name, var.name, input_type.type_str,
                            var.sym_type.__type_info__()))



class _InputType(object):
    """
    (Untyped) input containing fundamental properties of all inputs to an
    Operation:
    """

    def __init__(self, const=False, optional=False):
        """
        const (bool):
            True if the InputType has to be constant / materialized at compile time.
            Const InputType is semantically equivalent to attribute. By
            default False. Read-only.

        optional (bool):
            True to allow user not to specify this input and rely on default
            values (defined in default_inputs).

        Note: _InputType should not be directly instantiated. Only its subclasses may
        be instantiated.
        """
        self.const = const
        self.optional = optional

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

class FloatTensorInputType(ScalarOrTensorInputType):
    """
    Tensor input with float values
    with _sym_type in [types.fp16, types.fp32, types.fp64]

    Raise error when value set is not float.
    """

    def __init__(self, **kwargs):
        super(FloatTensorInputType, self).__init__(**kwargs)

    def _is_compatible(self, v):
        return types.is_tensor(v.sym_type) and v.dtype in SUPPORT_FLOAT_TYPES
    @property
    def type_str(self):
        return 'float tensor'

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
