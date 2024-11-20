#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import copy

import numpy as np

from coremltools import _logger as logger
from coremltools.converters.mil.mil import Block, get_existing_symbol, get_new_symbol, types
from coremltools.converters.mil.mil.input_type import (
    DefaultInputs,
    InputSpec,
    InternalInputType,
    ListInputType,
    PyFunctionInputType,
    TensorInputType,
    TupleInputType,
)
from coremltools.converters.mil.mil.operation import (
    NONE,
    SYMBOL,
    VALUE,
    Operation,
    mil_list,
    precondition,
)
from coremltools.converters.mil.mil.ops.defs._op_reqs import register_op
from coremltools.converters.mil.mil.ops.defs._utils import (
    infer_type_with_broadcast,
    promoted_primitive_type,
)
from coremltools.converters.mil.mil.types import is_compatible_type
from coremltools.converters.mil.mil.types.type_list import list as types_list
from coremltools.converters.mil.mil.types.type_mapping import (
    builtin_to_string,
    is_subtype,
    numpy_type_to_builtin_type,
    numpy_val_to_builtin_val,
)


@register_op
class cond(Operation):
    """
    Perform a conditional execution. The return types must be identical
    between the true and false branches.

    Parameters
    ----------
    pred: tensor<[], bool> (Required)
        * 0-D tensor (scalar) predicate to switch between true and false branches.

    _true_fn: function (Required)
        * A Python function that executes if ``pred`` evaluates to ``True``.
        * It must take zero input (i.e, no input), and return one or more values whose type becomes
          the operation's return type.

    _false_fn: function (Required)
        * A Python function that executes if ``pred`` evaluates to ``False``.
        * It must take zero input (i.e. no input), and have return types that match those of the
          ``if`` branch.

    _existing_blocks: list[Block] (Optional)
        * Python list of ``Block``.
        * For internal use only. When converting a milproto, we already got existing blocks,
          and the ``build_nested_blocks`` function can use them directly.
        * When ``_existing_blocks`` is set, ``_true_fn`` and ``_false_fn`` must be dummy functions which returns ``None``.

    Returns
    -------
    tuple
        * Python tuple of ``Variables`` from one of the branches.
    """

    input_spec = InputSpec(
        pred=TensorInputType(type_domain=types.bool),
        _true_fn=PyFunctionInputType(),
        _false_fn=PyFunctionInputType(),
        _existing_blocks=InternalInputType(optional=True),
    )

    def build_nested_blocks(self):
        # If the front end is milproto, we already have the well constructed cond/body block.
        # For this case, we set self.blocks directly.
        # We also check that _cond and _body are both dummy functions (return None).
        if self._existing_blocks is not None and self._existing_blocks.val is not None:
            assert self._true_fn.val([]) is None
            assert self._false_fn.val([]) is None
            self.blocks = self._existing_blocks.val
            return

        # Cond block
        true_block_name = self.name + "_true"
        with Block(name=true_block_name, outer_op=self) as true_block:
            true_func = self._true_fn.val
            true_ret_vars = true_func()
            if isinstance(true_ret_vars, tuple):
                true_ret_vars = list(true_ret_vars)
            if not isinstance(true_ret_vars, list):
                true_ret_vars = [true_ret_vars]
            true_block.set_outputs(true_ret_vars)
            self.blocks.append(true_block)

        false_block_name = self.name + "_false"
        with Block(name=false_block_name, outer_op=self) as false_block:
            false_func = self._false_fn.val
            false_ret_vars = false_func()
            if isinstance(false_ret_vars, tuple):
                false_ret_vars = list(false_ret_vars)
            if not isinstance(false_ret_vars, list):
                false_ret_vars = [false_ret_vars]
            false_block.set_outputs(false_ret_vars)
            self.blocks.append(false_block)

    def type_inference(self):
        true_ret_vars = self.blocks[0].outputs
        false_ret_vars = self.blocks[1].outputs
        # Verify true_ret_vars has the same types as false_ret_vars
        for i, (vt, vf) in enumerate(zip(true_ret_vars, false_ret_vars)):
            if not is_compatible_type(vt.sym_type, vf.sym_type):
                msg = (
                    "true branch output {} type {} mismatch false branch"
                    + " output type {}"
                )
                raise ValueError(msg.format(vt.name,
                    vt.sym_type.__type_info__(), vf.sym_type.__type_info__()))

        return tuple(v.sym_type for v in true_ret_vars)

    def value_inference(self):
        if self.pred.val is None:
            raise NotImplementedError()
        if self.pred.val:
            return [v.val for v in self.blocks[0].outputs]
        return [v.val for v in self.blocks[1].outputs]


class Const(Operation):
    """
    A base class that returns constant values.

    Parameters
    ----------
    val: const<\\*,T> (Required)

    mode: immediate_value, file_value (Optional)
        * Determines how the constant value is stored in the internal MIL format.
        * For large constants such as convolution weights, use ``file_value``.
        * For smaller-size constants such as values of a stride, use ``immediate_value``.

    Returns
    -------
    const<\\*,T>

    Attributes
    ----------
    T: fp16, fp32, i32, str, bool
    """

    input_spec = InputSpec(
        val=InternalInputType(const=True),
    )

    def __init__(self, **kwargs):
        super(Const, self).__init__(**kwargs)
        self._weight_id = None

    def type_inference(self):
        builtin_type, _ = self._get_type_val(self.val.val)
        return builtin_type

    def value_inference(self):
        _, val = self._get_type_val(self.val.val)
        return val

    def _get_type_val(self, value):
        int32_max = np.int32(np.iinfo(np.int32).max)
        int32_min = np.int32(np.iinfo(np.int32).min)

        if isinstance(value, (float, np.float64)):
            value = np.float32(value)
        elif isinstance(value, bool):
            pass
        elif isinstance(value, (int, np.int64)):
            if value > int32_max:
                value = int32_max
            elif value < int32_min:
                value = int32_min
            else:
                value = np.int32(value)
        elif isinstance(value, (tuple, list, np.ndarray)):
            value = np.array(value) if isinstance(value, (tuple, list)) else value
            if value.dtype in [np.uint64, np.int64]:
                logger.debug(
                    f"Downcast const op {self.name} data {builtin_to_string(numpy_type_to_builtin_type(value.dtype))} as int32"
                )
                value_clip_max = np.where(value > int32_max, int32_max, np.int32(value))
                value_clip_min = np.where(value_clip_max < int32_min, int32_min, np.int32(value_clip_max))
                value = value_clip_min
            if value.dtype == np.float64:
                logger.debug(f"Downcast const op {self.name} data fp64 as fp32")
                value = value.astype(np.float32)
        elif isinstance(value, mil_list):
            # If val that was passed in is of type mil_list, which is just a wrapper on top of
            # python list, then construct the list type.
            list_value = value.ls
            if len(list_value) == 0:
                raise ValueError("'mil_list' points to an empty list")
            builtin_elem_type, _ = self._get_type_val(list_value[0])
            # mil_list is a special case that we want to preserve the int64 element type
            if isinstance(list_value[0], np.int64):
                builtin_elem_type = types.int64
            builtin_type = types_list(builtin_elem_type, init_length=len(list_value), dynamic_length=False)
            return builtin_type, value

        if not isinstance(value, (np.generic, np.ndarray, str, bool, mil_list)):
            raise ValueError(f"Unknown value for constant: {value}")

        _, builtin_type = numpy_val_to_builtin_val(value)
        return builtin_type, value

    @property
    def weight_id(self) -> str:
        """
        Weight id for the const. It is used for weight sharing across multiple functions.
        Constants sharing the same weight_id will use the same blob file value when
        lowering to milproto.
        """
        return self._weight_id

    @weight_id.setter
    def weight_id(self, val: str) -> None:
        """
        Set weight id for the const.
        """
        assert isinstance(val, str), f"weight_id must be type of str. Got {type(val)}."
        assert self._weight_id is None, f"cannot set {self.name} weight_id twice."
        self._weight_id = val


@register_op
class const(Const):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


# Internal const can have symbolic value (for testing purpose)
@register_op
class _const_symbolic(const):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def type_inference(self):
        builtin_type, _ = self._get_type_val(self.val.sym_val)
        return builtin_type

    def value_inference(self):
        # We allow symbolic values in _const_symbolic
        _, val = self._get_type_val(self.val.sym_val)
        return val


@register_op
class select(Operation):
    """
    Return the elements selected from either ``a`` or ``b`` depending on the ``cond``.

    You must provide ``a``, ``b`` and ``cond``. The shape of ``cond``, ``a``,
    and ``b`` must be broadcastable.

    Parameters
    ----------
    cond: tensor<[\\*D1], B> (Required)
        * Tensor. When ``True``, select element from ``x``, otherwise, ``y``.

    a: tensor<[\\*D2], T> (Optional)
        * Values selected at indices where ``cond`` is ``True``.
        * Default is ``None``.

    b: tensor<[\\*D3], T> (Optional)
        * Values selected at indices where ``cond`` is ``False``.
        * Default is ``None``.

    Returns
    -------
    tensor<[\\*D_out], T> or tensor<[n, len(D1)], int32>
        *  If ``a, b`` are both provided, the return shape is based on broadcast rules
           from ``cond, a, b``.
        *  If ``a, b`` are ``None``, the return shape is 2-D, where the first dimension
           ``n`` is the number of matching indices in ``cond``, and ``len(D1)`` is the
           rank of ``cond``.

    Attributes
    ----------
    B: bool
    T: fp16, fp32, i32, bool
    """

    input_spec = InputSpec(
        cond=TensorInputType(type_domain=types.bool),
        a=TensorInputType(type_domain="T"),
        b=TensorInputType(type_domain="T")
    )

    type_domains = {
        "T": (types.fp16, types.fp32, types.bool, types.int32),
    }

    def type_inference(self):
        typea = self.a.sym_type
        typeb = self.b.sym_type
        primitive_type = promoted_primitive_type(typea, typeb)
        if primitive_type is None:
            raise ValueError("Incompatible primitive types in broadcast operation")

        return infer_type_with_broadcast(typea, typeb, primitive_type)

    @precondition(allow=VALUE)
    def value_inference(self):
        res = np.where(self.cond.val, self.a.val, self.b.val)
        sym_type = self.type_inference()
        if types.is_scalar(sym_type) and not np.isscalar(res):
            res = getattr(np, str(res.dtype))(res.item())
        return res


@register_op
class while_loop(Operation):
    """
    Perform the body repeatedly while the condition ``cond`` is true.

    Parameters
    ----------
    _cond: function  (Required)
        * A Python function that takes ``loop_vars`` as positional arguments.
        * The function must return a ``bool`` ``Var``.

    _body: function  (Required)
        * A Python function that takes ``loop_vars`` as positional arguments.
        * The function must return the same number of output vars as ``loop_vars``
          with the same types.

    loop_vars: tuple (Required)
        * Python tuple of ``Variables``.

    _existing_blocks: list[Block] (Optional)
        * Python list of ``Block``.
        * For internal use only. When converting a milproto, we already got existing blocks,
          and the ``build_nested_blocks`` function can use them directly.
        * When ``_existing_blocks`` is set, ``_cond`` and ``_body`` must be dummy functions which returns ``None``.

    Returns
    -------
    tuple
        * Python tuple (same type as ``loop_vars``).
    """

    input_spec = InputSpec(
        # arg name with underscore prefix won't be printed.
        _cond=PyFunctionInputType(),
        _body=PyFunctionInputType(),
        loop_vars=TupleInputType(),
        _existing_blocks=InternalInputType(optional=True),
    )

    @staticmethod
    def _check_equal_value(val1, val2):
        if val1 is None and val2 is None:
            return True
        if val1 is None or val2 is None:
            return False
        if isinstance(val1, np.ndarray) and isinstance(val2, np.ndarray):
            return np.array_equal(val1, val2)
        return val1 == val2

    @staticmethod
    def _clean_up_child_ops(block):
        for op in block.operations:

            for b in op.blocks:
                while_loop._clean_up_child_ops(b)

            inputs = op.get_flattened_inputs()
            for in_var in inputs:
                in_var.remove_child_op(op)

    def _build_block(self, block_inputs):
        # Cond block:
        block_name = self.name + '_cond_block'
        with Block(block_inputs=block_inputs, outer_op=self,
                   name=block_name) as cond_block:

            cond_func = self._cond.val
            cond_var = cond_func(*cond_block.inputs)
            cond_vars = cond_var if isinstance(cond_var, list) else [cond_var]
            cond_block.set_outputs(cond_vars)

        # Body block
        block_name = self.name + '_body_block'
        with Block(block_inputs=block_inputs, outer_op=self,
                   name=block_name) as body_block:
            body_func = self._body.val
            exit_vars = body_func(*body_block.inputs)
            exit_vars = list(exit_vars) if isinstance(exit_vars, (list, tuple)) \
                else [exit_vars]
            body_block.set_outputs(exit_vars)

        return cond_block, body_block, exit_vars

    def build_nested_blocks(self):
        # self.loop_vars is python tuple of Vars.

        # block_inputs Var are not produced by any op.
        # We assume block_inputs have the same types as self.loop_var. If not
        # (e.g., when certain dimensions change shape during iterate), we'd
        # adjust later.

        # We assume that sym_val is unchanging across the block iterate. If it
        # changes, we rebuild the block and rerun type and value inference.

        # Design notes on two blocks (cond and body):
        #
        # - Observe that two blocks can always be represented as a single
        # block that contains both cond and body logic, which would return
        # [loop_cond] + loop_carries. `loop_cond` is a bool.
        #
        # - Observe that single block implies a do-while logic,
        # in which the first iterate is always executed. It's possible to add
        # a cond input to while_loop to modify do-while behavior:
        #
        #   %first_cond = cond_logic(...)
        #   while_loop(cond=%first_cond, loop_vars=(...))
        #
        # and we enter the first iterate only if cond is True. But this would
        # require caller to execute cond logic outside of while_loop first
        # (which also needs to be duplicated within the loop),
        # resulting in duplicated code / ops.
        #
        # - Thus, single block is unnatural for the natural execution order,
        # in which we execute the cond block first to get the loop_cond. Only
        # if `loop_cond` is True do we execute the body block. This is the
        # semantics of tf.while_loop.

        # If the front end is milproto, we already have the well constructed cond/body block.
        # For this case, we set self.blocks directly.
        # We also check that _cond and _body are both dummy functions (return None).
        if self._existing_blocks is not None and self._existing_blocks.val is not None:
            assert self._cond.val([]) is None
            assert self._body.val([]) is None
            self.blocks = self._existing_blocks.val
            return

        block_inputs = tuple(copy.copy(v) for v in self.loop_vars)
        name_count = {v.name: 0 for v in block_inputs}
        for v in block_inputs:
            v._op = None
            v.op_output_idx = None
            v._child_ops = list()

            # Get unique name

            old_v_name = v.name
            v.name = v.name + "_x" + str(name_count[v.name])
            name_count[old_v_name] += 1

            v._sym_val = v._sym_val
            v.consuming_blocks = list()

        cond_block, body_block, exit_vars = self._build_block(block_inputs)

        # Verify exit_vars has the same types as loop_vars
        block_input_type_change = False
        for i, (v_in, v_out) in enumerate(zip(block_inputs, exit_vars)):
            if not is_subtype(v_out.sym_type, v_in.sym_type):
                compat_shape = while_loop.get_compat_shape(v_out.sym_type,
                        v_in.sym_type)
                if compat_shape is None:
                    msg = "loop_vars '{}' changes in the body of " \
                          "while_loop '{}':\n {} -> {}"
                    raise ValueError(msg.format(
                        v_in.name, self.name,
                        v_in.sym_type, v_out.sym_type))
                else:
                    block_inputs[i]._sym_type = types.tensor(
                            v_in.dtype, compat_shape)
                    block_input_type_change = True
            if not while_loop._check_equal_value(v_out.sym_val, v_in.sym_val):
                block_inputs[i]._sym_val = None
                block_input_type_change = True

        if block_input_type_change:
            # Since we are going to build the block again, we first need to remove ops
            # in the block from vars's _child_ops.
            while_loop._clean_up_child_ops(cond_block)
            while_loop._clean_up_child_ops(body_block)

            # Rebuild our block to invoke type inference.
            cond_block, body_block, exit_vars = self._build_block(block_inputs)
            for i, (v_in, v_out) in enumerate(zip(block_inputs, exit_vars)):
                if not is_subtype(v_out.sym_type, v_in.sym_type):
                    msg = 'Block output {}: {} is not a subtype of ' +\
                            'block input {}: {} after factoring shape changes'
                    raise ValueError(msg.format(v_out.name, v_out.sym_type.__name__,
                        v_in.name, v_in.sym_type.__name__))
                if not while_loop._check_equal_value(v_out.sym_val, v_in.sym_val):
                    msg = 'Block output {}: {} is not equal to ' +\
                            'block input {}: {} after value changes'
                    raise ValueError(msg.format(v_out.name, v.sym_val,
                        v_in.name, v_in.sym_val))
        self.blocks.append(cond_block)
        self.blocks.append(body_block)

    @staticmethod
    def get_compat_shape(type1, type2):
        """
        For tensor types `type1`, `type2` that are of the same rank, return
        compat_shape (python list) where compat_shape[i] is integer iff type1
        and type2 have the same integer shape on dim i. compat_shape[i] is
        symbolic otherwise.

        Return None if `type1`, `type2` have different rank or non-tensor
        type.
        """
        if not types.is_tensor(type1) or not types.is_tensor(type2):
            return None

        s1 = type1.get_shape()
        s2 = type2.get_shape()

        if len(s1) != len(s2):
            return None

        compat_shape = []
        for d1, d2 in zip(s1, s2):
            if d1 != d2:
                compat_shape.append(get_new_symbol())
            else:
                compat_shape.append(d1)
        return compat_shape

    def type_inference(self):
        # Skip the conditional var
        return tuple(v.sym_type for v in self.blocks[1].outputs)


@register_op
class make_list(Operation):
    """
    Create a list of tensor elements. The elements should have the same shape.
    The list is similar to an auto-resizing array.

    Parameters
    ----------
    init_length: <i32> (Optional, Default=1)
        * Initial length for the list.
        * If ``dynamic_length`` is ``False``,
          ``init_length`` is the fixed length of the list throughout runtime.

    dynamic_length: <bool> (Optional, Default is True)

    elem_shape: Tuple[const<T>] (Required)
        * 1-D vector denoting the shape of elements.
        * If ``T = int32``, the element shape is known at compile time.
        * ``T = string`` denotes the symbolic shape, in which the shape is determined
          at runtime.
        * If not provided, the resulting ``List`` won’t have the elementary shape
          info, which may cause backend errors. Remedy this with SSA passes.

    dtype: const (Optional, Default is fp32)
        * Possible values: ``{"bool", "fp16", "fp32", "int32"}``
        * Element tensor’s ``dtype``.

    Returns
    -------
    List[*]

    Attributes
    ----------
    T: i32, string
    """

    input_spec = InputSpec(
        init_length=TensorInputType(optional=True, type_domain=types.int32),
        dynamic_length=TensorInputType(const=True, optional=True, type_domain=types.bool),
        elem_shape=TupleInputType(),
        dtype=TensorInputType(const=True, optional=True, type_domain=types.str),
    )

    def default_inputs(self):
        return DefaultInputs(
            init_length=1,
            dynamic_length=True,
            dtype="fp32",
            )

    def type_inference(self):
        builtin_dtype = types.string_to_builtin(self.dtype.val)
        if builtin_dtype is None:
            raise ValueError("Unsupported dtype {}".format(self.dtype.val))
        # Replace string with symbol
        elem_shape_sym = []
        for s_var in self.elem_shape:
            # s is str or int
            s = s_var.val
            if s is None:
                msg = 'make_list elem_shape must be tuple of const. ' +\
                    'Tuple elem {} is not'
                raise ValueError(msg.format(s_var.name))

            if isinstance(s, str):
                try:
                    symbol = get_existing_symbol(s)
                except ValueError:
                    # Must be a new symbol
                    symbol = get_new_symbol(s)
                elem_shape_sym.append(symbol)
            else:
              elem_shape_sym.append(s)
        elem_type = types.tensor(builtin_dtype, elem_shape_sym)
        return types.list(
            elem_type,
            init_length=self.init_length.val,
            dynamic_length=self.dynamic_length.val,
        )


@register_op
class list_length(Operation):
    """
    Return the length of ``ls``.

    Parameters
    ----------
    ls: List[*] (Required)

    Returns
    -------
    <i32>
        * Length of ``ls``.

    Attributes
    ----------
    T: fp16, fp32, i32, bool
    """

    input_spec = InputSpec(ls=ListInputType(),)

    def type_inference(self):
        return types.int32

    @precondition(allow=VALUE | SYMBOL | NONE)
    def value_inference(self):
        if not self.ls.dynamic_length:
            return self.ls.init_length
        raise NotImplementedError()


@register_op
class list_write(Operation):
    """
    Write a value into index ``index`` of ``ls``.

    Parameters
    ----------
    ls: List (Required)

    index: <i32> (Required)
        * Size of the list.

    value: <*,T> (Optional)
        * Element value to write, which must match the element shape of ``ls``.
        * Default is ``None``.

    Returns
    -------
    List[*]

    Attributes
    ----------
    T: fp16, fp32, i32, bool
    """

    input_spec = InputSpec(
        ls=ListInputType(),
        index=TensorInputType(type_domain=types.int32),
        value=TensorInputType(type_domain="T"),
    )

    type_domains = {
        "T": (types.fp16, types.fp32, types.bool, types.int32),
    }

    def type_inference(self):
        list_elem_type = self.ls.elem_type
        value_type = self.value.sym_type
        dynamic_length = self.ls.dynamic_length
        init_length = self.ls.init_length

        if list_elem_type is None:
            # fill in the elem type using value's type info.
            return types.list(
                value_type, init_length=init_length, dynamic_length=dynamic_length
            )
        if list_elem_type == types.unknown:
            msg = "Input ls elem type unknown. Override with {}"
            logger.warning(msg.format(value_type))
            return types.list(
                value_type, init_length=init_length, dynamic_length=dynamic_length
            )
        if not types.is_subtype(value_type, list_elem_type):
            msg = "Elem type mismatch: ls elem type {} vs " + "value type {}"
            raise ValueError(msg.format(list_elem_type.__type_info__(),
              value_type.__type_info__()))
        return self.ls.sym_type


@register_op
class list_read(Operation):
    """
    Read the value at location ``index`` of ``ls``.

    Parameters
    ----------
    ls: List[\\*] (Required)

    index: <i32> (Required)
        * Size of the list.

    Returns
    -------
    <\\*,T>
        * The element's value.

    Attributes
    ----------
    T: fp16, fp32, i32, bool
    """

    input_spec = InputSpec(
        ls=ListInputType(),
        index=TensorInputType(type_domain=types.int32),
        )

    def type_inference(self):
        list_elem_type = self.ls.elem_type
        if list_elem_type is None:
            msg = (
                "Unknown element type. The List might not have been "
                + "written to ({})"
            )
            raise ValueError(msg.format(self.name))
        return list_elem_type


@register_op
class list_gather(Operation):
    """
    Return selected values in ``ls`` as a packed ``Tensor``.

    Parameters
    ----------
    ls: List[\\*] (Required)

    indices: <K, i32> (Required)
        * Gather from indices, whose element must be in ``[0, ls.length)`` at runtime.

    Returns
    -------
    <\\*K, T>
        * Selected tensors packed into a ``len(ls.elem_shape)+1`` rank tensor.
        * ``K[0] == len(indices)``.

    Attributes
    ----------
    T: fp16, fp32, i32, bool
    """

    input_spec = InputSpec(
        ls=ListInputType(),
        indices=TensorInputType(type_domain=types.int32),
        )

    def type_inference(self):
        list_elem_type = self.ls.elem_type
        if list_elem_type == types.unknown:
            msg = (
                "Unknown element type. The List might not have been "
                + "written to ({})"
            )
            raise ValueError(msg.format(self.name))
        elem_shape = list_elem_type.get_shape()
        dtype = list_elem_type.get_primitive()
        ret_shape = [self.indices.shape[0]] + list(elem_shape)
        return types.tensor(dtype, tuple(ret_shape))


@register_op
class list_scatter(Operation):
    """
    Scatter ``values`` to ``ls`` at locations ``indices``.

    Parameters
    ----------
    ls: List[*] (Required)

    indices: tensor<num_updates, i32> (Required)
        * Indices of ``ls`` to scatter to.
        * Elements of ``indices`` must be in ``[0, ls.length)`` at runtime.
        * If indices are greater than or equal to the list length, the list is
          dynamically resized.

    value: <*,T> (Optional)
        * Element value to write, which must match the element shape of ``ls``.
        * Default is ``None``.

    Returns
    -------
    List[*]
        * Updated list.

    Attributes
    ----------
    T: fp16, fp32, i32, bool
    """

    input_spec = InputSpec(
        ls=ListInputType(),
        indices=TensorInputType(type_domain=types.int32),
        value=TensorInputType(type_domain="T"),
    )

    type_domains = {
        "T": (types.fp16, types.fp32, types.bool, types.int32),
    }

    def type_inference(self):
        num_indices = self.indices.shape[0]
        num_values = self.value.shape[0]
        if num_values != num_indices:
            raise ValueError(
                "Cannot scatter {} values to {} indices".format(num_values, num_indices)
            )
        list_elem_type = self.ls.elem_type
        value_type = self.value.sym_type
        dynamic_length = self.ls.dynamic_length
        init_length = self.ls.init_length

        elem_type = types.tensor(value_type.get_primitive(), value_type.get_shape()[1:])
        if list_elem_type == types.unknown:
            # fill in the elem type using value's type info.
            return types.list(
                elem_type, dynamic_length=dynamic_length, init_length=init_length
            )
        if not types.is_subtype(elem_type, list_elem_type):
            msg = "Elem type mismatch: ls elem type {} vs " + "value type {}"
            raise ValueError(msg.format(list_elem_type, elem_type))
        return self.ls.sym_type
