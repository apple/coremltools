#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import copy
from collections import defaultdict
from typing import Dict, List, Optional, Union

import numpy as np

from coremltools.converters.mil.mil import types
from coremltools.converters.mil.mil.types.symbolic import any_symbolic

from .scope import ScopeSource


class Var:
    """
    Var represents the outputs of an Operation. Most Vars are derived from an
    Operation (including const), and all Vars must have `sym_type`.

    Example Usage:

    from coremltools.converters.mil.mil import (
        Builder as mb,
        Function,
        types
    )

    func_inputs = {"a": mb.placeholder(shape=(1,2)),
                   "b": mb.placeholder(shape=(1,2)) }
    with Function(func_inputs) as ssa_func:
        a, b = ssa_func.inputs["a"], ssa_func.inputs["b"]
        res = mb.add(x=a, y=b) # res is Var
        assert types.is_tensor(res.sym_type)
        assert res.rank == 2
        assert res.dtype == types.float # since a, b are by default float

        # value is not available at compile time in this  case. If
        # materializable, res.val would be a numpy / primitive value
        assert res.val is None


    Comment: Except InternalVar and Vars created in while_loop and by
    placeholder, all Var should only be constructed by Operation to represent
    outputs.

    Comment: Var hides the details of sym_type vs sym_val vs materialized
    value, which was represented by 2 objects prior to refactoring.


    # Properties:

    name: (str)
        name in MIL proto NamedValueType. Name is assigned by the parent
        Operation.

    sym_type [_sym_type]: (builtin type class)
        All Var must have a (possibly symbolic) type, usually derived from
        type inference of upstream ops or from default values in _Input.

    sym_val [_sym_val]: (builtin type instance)
        Possibly symbolic value.

    val [_sym_val]: (np.ndarray or python primitive scalar)
        Numpy (scalar / tensor) value. `val` is not None iff `sym_val` is
        not None and does not contain symbols.  Read-only.

    op [_op]: (Operation)
        The Operation this Var is derived from. May not be None except
        for InternalVar. Read-only.

    op_output_idx: (int)
        Idx of the output from Operation corresponding to _Input.  May be
        None.

    child_ops [_child_ops]: list[Operation]
        Ops that take this Var as an input.

    nonreplaceable_vars_upstream: set[Var]
        Set that consists of nonreplaceable vars upstream
    """

    __slots__ = [
        "name",
        "_sym_type",
        "_sym_val",
        "_op",
        "op_output_idx",
        "_child_ops",
        "consuming_blocks",
        "_nonreplaceable_vars_upstream",
        "is_descendant_of_const",
    ]

    def __init__(
        self,
        name,
        sym_type,
        sym_val=None,
        op=None,
        op_output_idx=None,
    ):
        """
        sym_type (builtin type)
        sym_val (builtin value)
        op (Operation)
        op_output_idx (int)
        """
        self.name = name
        self._sym_type = sym_type
        self._sym_val = sym_val
        self._op = op
        self.op_output_idx = op_output_idx
        # An op can appear twice if it consumes a var twice (e.g.,
        # add(%1, %1), while_loop(loop_vars=(%1, %1)).
        self._child_ops = list()

        # A variable may not be consumed by any op (i.e. len(self._child_ops)
        # == 0) but is still used as block output. A var can be output of
        # multiple blocks (e.g., both current block and nested blocks)
        self.consuming_blocks = list()

        # replaceability
        self._nonreplaceable_vars_upstream = set()
        self._set_nonreplaceable_vars_upstream()

        self._adjust_sym_val()

        # Track vars constness, which requires a var to satisfy one of the following:
        # 1. var.val is not None, which's mean the converter already has its compile time value through value inference.
        # 2. Is a descendant of ``constexpr_`` ops. We don't compute the value inference of those ``constexpr_`` ops,
        #    due to the fact it can potentially results in memory issue.
        self.is_descendant_of_const = Var._propagate_constness_upstream(self)

    def _adjust_sym_val(self):
        """For sub-byte dtype var, adjust the sym_val to make sure it reflects the true dtype."""
        if types.is_list(self.sym_type):
            return

        if not types.is_sub_byte(self.dtype):
            return

        if isinstance(self.sym_val, (np.generic, np.ndarray)):
            np_val = self._sym_val.val
            if (
                np_val.dtype.metadata is None
                or types.SUB_BYTE_DTYPE_METADATA_KEY not in np_val.dtype.metadata
            ):
                target_np_dtype = types.nptype_from_builtin(self.dtype)
                self._sym_val.val = np_val.astype(target_np_dtype)

    @property
    def nonreplaceable_vars_upstream(self):
        return self._nonreplaceable_vars_upstream

    @nonreplaceable_vars_upstream.setter
    def nonreplaceable_vars_upstream(self, val):
        assert isinstance(val, set)
        self._nonreplaceable_vars_upstream = val

    @staticmethod
    def _is_nonreplaceable_var(var):
        op = var.op
        if op is None:
            return False
        return op.op_type.startswith("constexpr_")

    @staticmethod
    def _propagate_constness_upstream(var):
        op = var.op
        if op is None:
            return False
        if (
            op.op_type.startswith("constexpr_")
            or (op.op_type == "dequantize" and op.can_materialize_val())
            or var.val is not None
        ):
            return True
        flattened_inputs = op.get_flattened_inputs()
        return all([x.is_descendant_of_const for x in flattened_inputs])

    def _set_nonreplaceable_vars_upstream(self):
        """
        A utility function to set the value of the "nonreplaceable_vars_upstream" property.
        If self is a non-replaceable var, then "nonreplaceable_vars_upstream" is a single element set, containing self.
        Otherwise, it is a union of the "nonreplaceable_vars_upstream" sets of all the input vars of its parent ops.
        """
        op = self.op
        if op is None:
            return
        if op.op_type == "shape":
            # For the meta data ops, like shape, we stop propogate the nonreplaceable_vars.
            self.nonreplaceable_vars_upstream = set()
            return
        if Var._is_nonreplaceable_var(self):
            self.nonreplaceable_vars_upstream = set([self])
        else:
            flattened_inputs = op.get_flattened_inputs()
            inputs_nonreplaceable_vars_upstream = [p.nonreplaceable_vars_upstream for p in flattened_inputs]
            if len(inputs_nonreplaceable_vars_upstream) > 0:
                self.nonreplaceable_vars_upstream = set.union(*inputs_nonreplaceable_vars_upstream)

    def _reset_nonreplaceable_vars_upstream(self):
        self.nonreplaceable_vars_upstream = set()

    def can_be_replaced_by_var(self, new_var):
        """
        A var can be replaced by a new var only if the new var's nonreplaceable_vars_upstream is the super set of the old one
        """
        return self.nonreplaceable_vars_upstream.issubset(new_var.nonreplaceable_vars_upstream)

    def can_be_folded_to_const(self) -> bool:
        """
        When translating frontend ops to PyMIL ops, some vars could be directly folded into a const.
        For example, in PyTorch's `to()` op, the input could be converted by `cast` op, or directly
        be folded to const.

        We only fold the var to a const when its value is known AND it doesn't have any
        non-replaceable vars in the upstream.
        """
        return self.val is not None and not self.nonreplaceable_vars_upstream

    @property
    def sym_type(self):
        return self._sym_type

    @property
    def shape(self):
        if types.is_tensor(self._sym_type):
            return self._sym_type.get_shape()
        if types.is_state(self._sym_type):
            wrapped_type = self._sym_type.wrapped_type()
            assert types.is_tensor(wrapped_type), "only tensor type is supported in state type."
            return wrapped_type.get_shape()
        return tuple()

    @property
    def rank(self):
        return len(self.shape)

    @property
    def dtype(self):
        if types.is_tensor(self._sym_type):
            return self._sym_type.get_primitive()
        if types.is_state(self._sym_type):
            wrapped_type = self._sym_type.wrapped_type()
            assert types.is_tensor(wrapped_type), "only tensor type is supported in state type."
            return wrapped_type.get_primitive()
        return self._sym_type

    @property
    def sym_val(self):
        if self._sym_val is None:
            return None
        return self._sym_val.val

    @property
    def val(self):
        if self._sym_val is None or any_symbolic(self._sym_val.val):
            return None
        return self._sym_val.val

    @property
    def op(self):
        return self._op

    @property
    def child_ops(self):
        return self._child_ops

    def add_child_op(self, new_op):
        self._child_ops.append(new_op)

    def remove_child_op(self, target_op, no_check=False):
        if target_op not in self._child_ops:
            if no_check:
                return  # no-op
            msg = "Op {} does not takes Var {} as input"
            raise ValueError(msg.format(target_op.name, self.name))
        self._child_ops.remove(target_op)

    def shape_str(self):
        annotation = ""
        if self.val is not None:
            annotation = "*"
        elif self.sym_val is not None:
            annotation = "^"
        shape_str = str(self.shape)[:-1]  # trim the ")"
        if self.rank > 1:
            shape_str += ", "
        if types.builtin_to_string(self.dtype) is None:
            shape_str += ")" + annotation
        else:
            shape_str += types.builtin_to_string(self.dtype) + ")" + annotation
        return shape_str

    def type_str(self):
        is_tensor = types.is_tensor(self.sym_type)
        is_list = types.is_list(self.sym_type)
        is_state = types.is_state(self.sym_type)
        if is_tensor:
            type_string = "(Tensor)"
        elif is_list:
            type_string = "(List)"
        elif is_state:
            type_string = "(State)"
        else:
            type_string = "(Scalar)"
        return type_string

    def set_name(self, name):
        self.name = name

    def is_tensor_or_scalar_of(self, dtype: Union[str, type]):
        if isinstance(dtype, type):
            dtype = types.builtin_to_string(dtype)
        return (
            types.is_tensor(self.sym_type) or types.is_scalar(self.sym_type)
        ) and types.builtin_to_string(self.dtype) == dtype

    def __str__(self):
        return "%" + self.name + ": " + self.shape_str() + self.type_str()

    @property
    def scopes(self) -> Dict[ScopeSource, List[str]]:
        if self.op is None:
            # An empty dictionary is returned for function input vars.
            return defaultdict(list)
        return self.op.scopes

    @scopes.setter
    def scopes(self, scopes: Dict[ScopeSource, List[str]]):
        if self.op is None:
            raise ValueError(f"Cannot set scopes to a function input var {self}.")
        self.op.scopes = copy.deepcopy(scopes)


class ListVar(Var):
    __slots__ = ["_elem_type", "init_length", "dynamic_length"]

    def __init__(
        self, name, elem_type=None, init_length=None, dynamic_length=True, sym_val=None, **kwargs
    ):
        """
        elem_type (builtin.tensor)

        init_length (int): initial length

        dynamic_length (bool): True to allow list to grow. False uses
        init_length as the fixed size (init_length is runtime length).

        sym_val: value of the list, if available
        """
        super().__init__(
            name=name,
            sym_type=types.list(elem_type, init_length, dynamic_length),
            sym_val=sym_val,
            **kwargs
        )
        self._elem_type = elem_type
        self.init_length = init_length
        self.dynamic_length = dynamic_length

    @property
    def shape(self):
        raise ValueError("shape not applicable to ListVar '{}'.".format(self.name))

    @property
    def rank(self):
        raise ValueError("rank not applicable to ListVar '{}'".format(self.name))

    @property
    def dtype(self):
        raise ValueError("dtype not applicable to ListVar '{}'".format(self.name))

    @property
    def elem_type(self):
        return self._elem_type

    @property
    def elem_shape(self):
        if self._elem_type == types.unknown:
            return None
        elif types.is_tensor(self._elem_type):
            return self._elem_type.get_shape()
        return ()

    def shape_str(self):
        length = "?"
        if not self.dynamic_length:
            length = str(self.init_length)
        if self._elem_type == types.unknown:
            return "List[{}, unknown]".format(length)
        if self._elem_type == types.str:
            return "List[{}, str]".format(length)
        elif self._elem_type == types.int64:
            return "List[{}, int]".format(length)
        else:
            elem_shape = self._elem_type.get_shape()
            elem_dtype = self._elem_type.get_primitive()
            shape_str = str(elem_shape)[:-1]  # trim the ")"
            if len(elem_shape) > 1:
                shape_str += ", "
            shape_str += types.builtin_to_string(elem_dtype) + ")"
            return "List[{}, {}]".format(length, shape_str)


class InternalVar(Var):
    """
    Internal Var (with '__' prefix and won't appear in SSA) will ALWAYS have
    `sym_val == builtin.unknown`. InternalVar are constructed by builder only.

    Comment: Internal Var can be used to represent diverse types such as enum
    type `DataType.FLOAT32`.
    """

    def __init__(self, val, name=None):
        super().__init__(
            name=name, sym_type=types.unknown, sym_val=types.unknown(val)
        )


class ComplexVar(Var):
    """Var to handle complex data."""

    __slots__ = ["_real", "_imag"]

    def __init__(
        self,
        name,
        sym_type,
        sym_val=None,
        op=None,
        op_output_idx=None,
        real: Optional[Var] = None,
        imag: Optional[Var] = None,
    ):
        super().__init__(
            name=name,
            sym_type=sym_type,
            sym_val=sym_val,
            op=op,
            op_output_idx=op_output_idx,
        )

        # Handle complex data types.
        self._real: Optional[Var] = real
        self._imag: Optional[Var] = imag

    @property
    def real(self):
        return self._real

    @property
    def imag(self):
        return self._imag

    @real.setter
    def real(self, real):
        if not types.is_complex(self.dtype):
            raise ValueError(
                f"Only complex number can set `real`. This var is {self.dtype}."
            )
        self._real = real

    @imag.setter
    def imag(self, imag):
        if not types.is_complex(self.dtype):
            raise ValueError(
                f"Only complex number can set `imag`. This var is {self.dtype}."
            )
        self._imag = imag
