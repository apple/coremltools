#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from collections import OrderedDict

from coremltools.converters.mil.mil import types
from coremltools.converters.mil.mil.var import InternalVar

SUPPORT_FLOAT_TYPES = [
    types.fp16,
    types.fp32,
    types.fp64,
]

SUPPORT_INT_TYPES = [
    types.uint8,
    types.uint16,
    types.uint32,
    types.uint64,
    types.int8,
    types.int16,
    types.int32,
    types.int64,
]

SUPPORT_COMPLEX_TYPES = [
    types.complex64,
    types.complex128,
]

_SUPPORT_TYPES = (
    SUPPORT_FLOAT_TYPES
    + SUPPORT_INT_TYPES
    + SUPPORT_COMPLEX_TYPES
    + [types.bool, types.str]
)


class DefaultInputs:
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
        new_order_dict = {k: v for k, v in self._ordered_dict.items()}
        for k, v in default_inputs._default_inputs:
            new_order_dict[k] = v
        return DefaultInputs(**new_order_dict)


class InputSpec:
    def __init__(self, **kwargs):
        # Since python 3.6, kwargs preserves the input order. See
        # https://docs.python.org/3/whatsnew/3.6.html#whatsnew36-pep468
        self._input_types = [(k, v) for k, v in kwargs.items()]
        self._ordered_dict = OrderedDict()
        for k, v in self._input_types:
            self._ordered_dict[k] = v

    def __add__(self, input_spec):
        new_order_dict = {k: v for k, v in self._ordered_dict.items()}
        for k, v in input_spec._input_types:
            new_order_dict[k] = v
        return InputSpec(**new_order_dict)


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

        # check vars sharing the same type_domain_id have the same dtype
        type_domain_group = {}
        var_to_input_name = {}
        for name, var in candidate_kvs.items():
            input_type = self.input_types[name]
            if isinstance(input_type, TensorInputType) and input_type.type_domain_id is not None:
                type_domain_id = input_type.type_domain_id
                if type_domain_id in type_domain_group:
                    type_domain_group[type_domain_id].append(var)
                else:
                    type_domain_group[type_domain_id] = [var]
                var_to_input_name[var] = name

        for type_domain_id, vars in type_domain_group.items():
            expected_dtype = vars[0].dtype
            ref_name = var_to_input_name[vars[0]]
            for var in vars:
                name = var_to_input_name[var]
                if not var.dtype == expected_dtype:
                    msg = (
                        "In op, of type {}, named {}, the named input `{}` must have the same data type "
                        "as the named input `{}`. However, {} has dtype {} whereas {} has dtype {}."
                    ).format(op_type, op_name, name, ref_name, name,
                             var.dtype.__type_info__(), ref_name, expected_dtype.__type_info__())
                    raise ValueError(msg)

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
            if (
                input_type.const
                and not isinstance(input_type, InternalInputType)
                and not var.is_descendant_of_const
            ):
                msg = msg_prefix + "Input {} must be const at compile time"
                raise ValueError(msg.format(name), name, var.name)

            if not isinstance(var, InternalVar) and \
                not input_type.is_compatible(var):
                msg = msg_prefix + "Input {}=\"{}\" expects " +\
                        "{} but got {}"
                raise ValueError(msg.format(name, var.name, input_type.type_str,
                            var.sym_type.__type_info__()))


class _InputType:
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


class TensorInputType(_InputType):
    """
    TensorInputType specifies the generic tensor inputs.
    The `type_domain` validates data type constraints, and it could be either
    (1) A object / tuple of builtin types:
        This puts constraint on the allowed inputs data type.
        For example:

        ```
        input_spec = InputSpec(
           x=TensorInputType(type_domain=types.int32),
        )
        ```
        only allows input `x` have int32 dtype.

        ```
        input_spec = InputSpec(
           x=TensorInputType(type_domain=(types.int32, types.fp16)),
        )
        ```
        allows input `x` be either type of int32 or float16

    (2) string:
        Verify different input parameters binding with the same `type_domain` are the same data type.
        This additional check is done by defining a `type_domains` dictionary in the Operation class
        For example:

        ```
        class conv(Operation):
            input_spec = InputSpec(
                x=TensorInputType(type_domain="T"),
                weight=TensorInputType(type_domain="U"),
            )

            type_domains = {
                "T": (types.fp16, types.fp32),
            }
        ```
        would verify:
        (i) `x` and `weight` are one of the float16 or float32 type.
        (ii) `x` and `weight` are the same type.

    """
    def __init__(self, type_domain, **kwargs):
        self._type_domain = ()
        self._type_domain_id = None

        if isinstance(type_domain, str):
            self.type_domain_id = type_domain
        else:
            if isinstance(type_domain, type):
                type_domain = (type_domain,)
            self.type_domain = type_domain
        super().__init__(**kwargs)

    def _is_compatible(self, v):
        result = types.is_scalar(v.dtype) or types.is_tensor(v.dtype)
        result = result and (v.dtype in self.type_domain)
        return result

    @property
    def type_domain(self):
        return self._type_domain

    @type_domain.setter
    def type_domain(self, val):
        msg = f"type_domain {val} must be a tuple of builtin types"
        if not isinstance(val, tuple) or any(map(lambda t: t not in _SUPPORT_TYPES, val)):
            raise ValueError(msg)
        self._type_domain = val

    @property
    def type_domain_id(self):
        return self._type_domain_id

    @type_domain_id.setter
    def type_domain_id(self, val):
        if not isinstance(val, str):
            raise ValueError("type_domain_id must be type of str")
        self._type_domain_id = val

    @property
    def type_str(self):
        return 'tensor or scalar of dtype from type domain ' + str([types.builtin_to_string(v) for v in self.type_domain])

class ListInputType(_InputType):
    """
    ListInputType allows inputs of type types.list
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _is_compatible(self, v):
        return types.is_list(v.sym_type)

    @property
    def type_str(self):
        return 'list'


class ListOrTensorInputType(_InputType):
    """
    ListOrTensorInputType allows inputs of
    (1) MIL tensor
    (2) python list/tuple of MIL tensors
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _is_compatible(self, v):
        return (
            types.is_list(v.sym_type)
            or types.is_scalar(v.dtype)
            or types.is_tensor(v.dtype)
        )

    @property
    def type_str(self):
        return 'list, tensor, or scalar'


class TupleInputType(_InputType):
    """
    TupleInputType specifies input types of python list/tuple of MIL tensors.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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
        super().__init__(**kwargs)

    def _is_compatible(self, v):
        return True  # skip type check by default for InternalInputType.


class PyFunctionInputType(InternalInputType):
    """
    Native python function.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _is_compatible(self, v):
        return callable(v.val)
