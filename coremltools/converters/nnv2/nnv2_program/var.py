# -*- coding: utf-8 -*-

from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _

import abc
ABC = abc.ABCMeta('ABC', (object,), {}) # compatible with Python 2 *and* 3

class Var(ABC):
    """
    Var is the interface type of inputs and outputs of an Operation. In single static assignment
    form (SSA), each Var is an unique instance.

    Var enhances builtin types with book keeping of the operations that input
    and output the values.  The main interface for Var are two sets of
    operations: accessing parent/children ops, and accessing type or value.

    Var holds the type information through `var.type : builtin_type class` filled by `op.type_inference`
    and optional value information through `var.val builtin_type value` filled by `op.value_inference`.
    Because the value inference is opportunistic, var.value may be None.

    The preferred way to extract value from a Var is the `get_value()`
    function, which returns the numpy value stored within the builtin
    value. `ScalarVar`, and `TensorVar` extends the `get_value(allow_symbolic)` to support
    optionally extracting values with symbols.

    On the other hand, the `val` property returns the underlying builtin
    typed value.

    Implementations:
        ScalarVar(Var, _SymbolicMixin)
        TensorVar(Var, _SymbolicMixin)
        ListVar (TODO)
    """
    @property
    @abc.abstractmethod
    def name(self):
        "Name of the Var"

    @property
    @abc.abstractmethod
    def op(self):
        "Operation that output this"

    @property
    @abc.abstractmethod
    def child_ops(self):
        """
        List[Operation] that take this as input.
        """

    def add_child_op(self, new_op):
        """
        Add an operation to the child_ops.
        """
        self.child_ops.add(new_op)

    def remove_child_op(self, target_op):
        """
        Remove an operation from child_ops.

        ValueError is raised if `target_op` does not take this as input.
        """
        if target_op not in self.child_ops:
            msg = "Op {} does not takes Var {} as input"
            raise ValueError(msg.format(target_op.name, self.name))
        self.child_ops.remove(target_op)


    @property
    def type(self):
        "builtin type class"
        pass

    @property
    def val(self):
        "Optional[builtin typed value]"
        pass

    @abc.abstractmethod
    def get_value(self):
        """ Access the pure value (if any) of the Var.

        Return:
            ret: builtin_val.val or None
        """

    @abc.abstractmethod
    def _type_str(self):
        "Return str representation of type"
        pass

    def __str__(self):
        return "%" + self.name + ": " + self._type_str()


class _SymbolicMixin(ABC):
    """
    Interface for Var containing symbolic value
    """
    @property
    @abc.abstractmethod
    def val(self):
        "Optional[builtin typed value or Symbol]"
        pass

    @abc.abstractmethod
    def get_value(self, allow_symbolic=False):
        """ Access the pure value (if any) of the Var.

        Args:
            allow_symbolic (bool): If false and the value is symbolic,
                return None. Useful when operation cannot handle symbolic
                value.

        Return:
            ret: builtin_val.val or None
        """

    @abc.abstractmethod
    def has_symbolic(self):
        "Return True if value exists and has symbol"
