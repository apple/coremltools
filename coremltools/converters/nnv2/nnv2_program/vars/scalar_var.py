# -*- coding: utf-8 -*-

from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _

from attr import attrs, attrib

from coremltools.converters.nnv2.builtin_types import builtins
from coremltools.converters.nnv2.nnv2_program.program import Operation
from coremltools.converters.nnv2.nnv2_program.var import Var, _SymbolicMixin
from coremltools.converters.nnv2.builtin_types.symbolic import is_symbolic


@attrs(frozen=True, slots=True)
class ScalarVar(Var, _SymbolicMixin):
    """Var with scalar value

    ScalarVar value is primitive: bool, int, float, str.
    ScalarVar value can be symbolic.

    Example:

        >>> from coremltools.converters.nnv2.builtin_types import builtins
        >>> from sympy import symbols
        >>> op = "dummy"

        >>> print(ScalarVar("x", op, builtins.int, builtins.int(8)))
        %x: (i64)*

        >>> print(ScalarVar("x", op, builtins.int, val=None))
        %x: (i64)

        >>> print(ScalarVar("x", op, builtins.int, val=symbols("x")))
        %x: (i64)^
    """

    name = attrib(type=str)
    op = attrib(type=Operation)
    type = attrib()                        # builtin type; cannot be None
    val = attrib()                         # optional builtin value of the same builtin type
    child_ops = attrib(factory=list)

    @type.validator
    def _check_type_primitive(self, attribute, value):
        if not builtins.is_primitive(value):
            raise TypeError("Expect type is primitive. Got {}".format(value))

    @val.validator
    def _check_value_builtin_or_none(self, attribute, value):
        if not (value is None or value.__class__ == self.type or is_symbolic(value)):
            raise TypeError("Expect value is None or {} or symbol. Got \
                            {}".format(self.type, value.__class__))

    def _type_str(self):
        if self.has_symbolic():
            annotation = "^"
        elif self.val is not None:
            annotation = "*"
        else:
            annotation = ""
        return "({}){}".format(builtins.builtin_to_string(self.type), annotation)

    def get_value(self, allow_symbolic=False):
        if self.val is None:
            return None
        elif self.has_symbolic():
            return self.val if allow_symbolic else None
        else:
            return self.val.val

    def has_symbolic(self):
        if self.val:
            return is_symbolic(self.val)
        else:
            return False
