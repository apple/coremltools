from coremltools.converters.nnv2.builtin_types import builtins
from coremltools.converters.nnv2.builtin_types.symbolic import (
        is_symbolic,
        any_symbolic,
        )
from .type_utils import builtin_to_str

class Var(object):
    """
    Var represents the outputs of an Operation. Most Vars are derived from an
    Operation (including const), and all Vars must have `sym_type`.

    Example Usage:

    from coremltools.converters.nnv2.nnv2_program.ops import CoremlBuilder as cb

    func_inputs = {"a": cb.placeholder(shape=(1,2)),
                   "b": cb.placeholder(shape=(1,2)) }
    with SsaFunction(func_inputs) as ssa_func:
        a, b = ssa_func.inputs["a"], ssa_func.inputs["b"]
        res = cb.add(x=a, y=b) # res is Var
        assert builtins.is_tensor(res.sym_type)
        assert res.rank == 2
        assert res.dtype == builtins.float # since a, b are by default float

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
        name in NNv2 proto NamedValueType. Name is assigned by the parent
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

    child_ops [_child_ops]: set[Operation]
        Ops that take this Var as an input.
    """
    __slots__ = [
        "name", "_sym_type", "_sym_val", "_op", "op_output_idx", "_child_ops"
    ]

    def __init__(self,
                 name,
                 sym_type,
                 sym_val=None,
                 op=None,
                 op_output_idx=None):
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
        self._child_ops = set()

    @property
    def sym_type(self):
        return self._sym_type

    @property
    def shape(self):
        if builtins.is_tensor(self._sym_type):
            return self._sym_type.get_shape()
        return tuple()

    @property
    def rank(self):
        return len(self.shape)

    @property
    def dtype(self):
        if builtins.is_tensor(self._sym_type):
            return self._sym_type.get_primitive()
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
        self._child_ops.add(new_op)

    def remove_child_op(self, target_op):
        if target_op not in self._child_ops:
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
        shape_str += builtin_to_str[self.dtype] + ")" + annotation
        return shape_str

    def __str__(self):
        return "%" + self.name + ": " + self.shape_str()


class TupleVar(Var):
    """
    self._elems: python tuple[Var]
    """
    __slots__ = ["_elems"]

    def __init__(self, name, elems, op=None, op_output_idx=None):
        """
        elems: python tuple[Var]
        """
        sym_type = builtins.tuple(tuple(e.sym_type for e in elems))
        super(TupleVar, self).__init__(name=name,
                                       sym_type=sym_type,
                                       sym_val=None)
        self._elems = elems

    @property
    def val(self):
        return tuple(e.val for e in self._elems)

    @property
    def shape(self):
        raise ValueError("shape not applicable to TupleVar {}".format(
            self.name))

    @property
    def rank(self):
        raise ValueError("rank not applicable to TupleVar {}".format(
            self.name))

    @property
    def dtype(self):
        raise ValueError("dtype not applicable to TupleVar {}".format(
            self.name))

    def __getitem__(self, index):
        if index >= len(self._elems):
            msg = "Index {} out of bound (TupleVar len: {})"
            raise ValueError(msg.format(index, len(self._elems)))
        return self._elems[index]

    def py_tuple(self):
        return self._elems

    def shape_str(self):
        return "(" + ", ".join(e.shape_str() for e in self._elems) + ")"

    def __str__(self):
        return "%" + self.name + ": " + self.shape_str()


class ListVar(Var):
    __slots__ = ["_elem_type"]

    def __init__(self, name, elem_type):
        super(ListVar, self).__init__(name=name,
                                      sym_type=builtins.list(elem),
                                      sym_val=None)
        self._elem_type = elem_type

    @property
    def shape(self):
        raise ValueError("shape not applicable to ListVar {}.".format(
            self.name))

    @property
    def rank(self):
        raise ValueError("rank not applicable to ListVar {}".format(self.name))

    @property
    def dtype(self):
        raise ValueError("dtype not applicable to ListVar {}".format(
            self.name))

    @property
    def elem_type(self):
        return self._elem_type


class InternalVar(Var):
    """
    Internal Var (with '__' prefix and won't appear in SSA) will ALWAYS have
    `sym_val == builtin.unknown`. InternalVar are constructed by builder only.

    Comment: Internal Var can be used to represent diverse types such as enum
    type `DataType.FLOAT32`.
    """
    def __init__(self, val, name=None):
        super(InternalVar, self).__init__(name=name,
                                          sym_type=builtins.unknown,
                                          sym_val=builtins.unknown(val))
