# -*- coding: utf-8 -*-

from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _

from attr import attrs, attrib

from coremltools.converters.nnv2.builtin_types import builtins
from coremltools.converters.nnv2.nnv2_program.program import Operation
from coremltools.converters.nnv2.nnv2_program.var import Var, _SymbolicMixin
from coremltools.converters.nnv2.builtin_types.symbolic import (is_symbolic,\
                                                                any_symbolic,
                                                                any_variadic)


@attrs(frozen=True, slots=True)
class TensorVar(Var, _SymbolicMixin):
    """Var with tensor value

    TensorVar has additional properties: dtype, rank, and shape.
    TensorVar value can be symbolic, e.g. (H/2, W/2, C*3)
    TensorVar supports 0D Tensor (Scalar).

    Example:

        >>> import numpy as np
        >>> from coremltools.converters.nnv2.builtin_types.builtins import type_mapping
        >>> from sympy import symbols
        >>> val = np.random.rand(5,5)
        >>> tensor_val, tensor_t = type_mapping.numpy_val_to_builtin_val(val)
        >>> op = "dummy"

        >>> print(TensorVar("x", op, tensor_t, tensor_val))
        %x: (5, 5, fp64)*

        >>> print(TensorVar("x", op, tensor_t, val=None))
        %x: (5, 5, fp64)

        >>> print(TensorVar("x", op, tensor_t, val=symbols("x1")))
        %x: (5, 5, fp64)^
    """

    name = attrib(type=str)
    op = attrib(type=Operation)
    type = attrib()                        # builtin type; cannot be None
    val = attrib()                         # optional builtin typed value of the same builtin type
    sym_val = attrib(default=None)         # optional symbolic value
    child_ops = attrib(factory=list)

    @type.validator
    def _check_type_tensor(self, attribute, value):
        if not builtins.is_tensor(value):
            raise TypeError("Expect type is tensor. Got {}".format(value))

    @val.validator
    def _check_value_builtin_or_none(self, attribute, value):
        if not (value is None or value.__class__ == self.type):
            raise TypeError("Expect value is None or {}. Got \
                            {}".format(self.type, value.__class__))

    @property
    def dtype(self):
        return self.type.get_primitive()

    @property
    def shape(self):
        """Return shape of the tensor type. Shape may contains symbolic value.
        """
        return self.type.get_shape()

    @property
    def rank(self):
        """Return the rank of the tensor type. If the shape is symbolic and
        contains variadic symbol, i.e (128, 128, *L), return None.
        """
        return None if any_variadic(self.shape) else len(self.shape)

    def _type_str(self):
        annotation = ""
        if self.has_symbolic():
            annotation = "^"
        elif self.val is not None:
            annotation = "*"
        else:
            annotation = ""
        type_str = str(self.shape)[:-1]  # trim the ")"
        if len(self.shape) > 1:
            type_str += ", "
        type_str += builtins.builtin_to_string(self.dtype) + ")" + annotation
        return type_str

    def get_value(self, allow_symbolic=False):
        if self.val is None:
            return None
        elif self.has_symbolic():
            return self.val.val if allow_symbolic else None
        else:
            return self.val.val

    def has_symbolic(self):
        if self.val:
            return is_symbolic(self.val) or any_symbolic(self.val.val)
        else:
            return False
