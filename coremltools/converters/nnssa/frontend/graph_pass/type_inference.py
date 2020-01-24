# -*- coding: utf-8 -*-
from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _

import collections
import logging
import math
import numpy as np
import operator
import sympy as sm
import sys
import six

PY3 = False
if sys.version_info >= (3, 0):
    PY3 = True

from ...commons import builtins
from ...commons.builtins.utils import promote_types
from ...commons.symbolic import *  # pylint: disable=wildcard-import

short_var_name_cache = {}
"""Mapping of operator names to a function to evaluate its symbolic_value."""
_SYMBOL_MAP_OPS = {
    'Add': operator.add,
    'Equal': operator.eq,
    'FloorDiv': operator.floordiv,
    'FloorMod': operator.mod,
    'Greater': operator.gt,
    'GreaterEqual': operator.ge,
    'Less': operator.lt,
    'LessEqual': operator.le,
    'Mul': operator.mul,
    'NotEqual': operator.ne,
    'RealDiv': operator.truediv,
    'Sub': operator.sub
}

_SYMBOL_REDUCE_OPS = {
    'ArgMax': np.argmax,
    'Max': np.amax,
    'Mean': np.mean,
    'Min': np.amin,
    'Prod': np.prod,
    'Sum': np.sum
}


def get_conv_outdim(in_dim, ks, stride, dl, padding_type):
    try:
        if padding_type == 'VALID':
            ks_dilated = (ks - 1) * dl + 1
            return (in_dim - ks_dilated) / stride + 1
        elif padding_type == 'SAME':
            return math.ceil(in_dim * 1.0 / stride)
        else:
            raise ValueError('[TypeInference] Unrecognized padding type.')
    except Exception as e:
        raise ValueError('[TypeInference] Error fetching padding values: {}'.format(e))


def get_short_var_name(name):
    if name in short_var_name_cache:
        return short_var_name_cache[name]
    else:
        shortname = 's_' + str(len(short_var_name_cache))
        short_var_name_cache[name] = shortname
        return shortname


def replace_neg_1_with_symbolic(val, name):
    for i in range(len(val)):
        if np.isscalar(val[i]) and val[i] == -1:
            val[i] = sm.Symbol(get_short_var_name(name + '_' + str(i)), positive=True)
    return val


def make_symbol(name):
    return sm.Symbol(get_short_var_name(name), positive=True)


def to_int(ls):
    for i in range(len(ls)):
        if is_symbolic_or_unknown(ls[i]):
            continue
        ls[i] = int(ls[i])
    return ls


def try_to_np_type(v):
    """
    np types are easier to handle than python primitive. (e.g., calling
    reshape on np.int32 is valid, but not on int.)
    """
    if isinstance(v, int):
        return np.int32(v)
    if isinstance(v, float):
        return np.float32(v)
    return v


def reshape_with_symbol(v, shape):
    """
    Perform basic reshape if v is symbolic (not array of symbols).
    """
    if is_symbolic_or_unknown(v):
        return np.array(v).reshape(shape)
    shape = [int(s) for s in shape]
    return v.reshape(shape)


def try_get_non_sym_val(node):
    """
    node: ssa node name
    Return None if `node` doesn't have fully materialized value, else the
    value.
    """
    if "symbolic_value" in node.attr and \
            node.attr["symbolic_value"] is not None and \
            node.attr["symbolic_value"].val is not None and \
            not is_symbolic_or_unknown(node.attr["symbolic_value"].val):
        return node.attr["symbolic_value"].val
    return None


class TypeInferenceVisitor(object):
    def __init__(self, graph, whole_ssa, pedantic=False):
        """
        Args:
            graph (dict): A mapping of node names to TFNodes representing the function
                to type check.
            whole_ssa (NetworkEnsemble): The program being compiled.
            pedantic (bool): If true, require consistency in tensor primitive types. When
                possible, this should be enabled. This is a more stringent standard than
                can currently be met by some our frontends. Therefore, we leave it off by
                default.
        """
        # the whole ssa is needed to propagation function calls
        self.op_rules = {}
        self.gdict = graph
        self.whole_ssa = whole_ssa
        self.visited = {}
        self.pedantic = pedantic

    def visit(self, node):
        # make sure node is a ParsedNode
        from ...nnssa import ParsedNode
        if not isinstance(node, ParsedNode):
            node = self.gdict[node]

        # do we already know the answer?
        if node.datatype is not None and not node.op.startswith('TensorArray'):
            # if it is a fully specified type, we just return it
            # if we seen it this round, we return it
            # otherwise we recurse
            if not builtins.is_tensor(node.datatype) or \
                   builtins.tensor_has_complete_shape(node.datatype) or \
                   node.name in self.visited:
                return node.datatype
        # look for the op's visit method
        method = 'visit_' + node.op
        visitor = getattr(self, method, None)
        if visitor is None:
            logging.warning('WARNING [TypeInference]: Op {} not implemented. Inferring shape from node attribute!'.format(node.op))
            visitor = self._get_type_from_attr

        # find the type of the node
        ret = None
        try:
            ret = visitor(node)
        except Exception as e:  # pylint: disable=broad-except
            logging.exception("[TypeInference] Failed to infer type of %s:%s", node.name, node.op)
            raise

        if ret is not None:
            self.visited[node.name] = 1
            node.datatype = ret
        else:
            logging.error("[TypeInference] Unable to infer type of node %s (%s)", node.name, node.op)
        return ret

    def visit_all(self):
        for i in self.gdict:
            self.visit(self.gdict[i])

    def _get_type_from_attr(self, node):
        if node.datatype is not None:
            return node.datatype

        node.parse_from_attr()
        if builtins.is_tensor(node.datatype):
            s = list(node.datatype.get_shape())
            for i in range(len(s)):
                if s[i] == -1:
                    s[i] = make_symbol(node.name + '_' + str(i))
            node.datatype = builtins.tensor(node.datatype.get_primitive(), tuple(s))
        return node.datatype

    def match_shape(self, shapea, shapeb):
        if len(shapea) != len(shapeb):
            return False
        for idx in range(len(shapea)):
            if shapea[idx] != shapeb[idx] and shapea[idx] > 0 and shapeb[idx] > 0:
                return False
        return True

    def strict_shape(self, typea, typeb):
        shape = list(typea.get_shape())
        shapeb = typeb.get_shape()
        for idx in range(len(shape)):
            if is_symbolic_or_unknown(shape[idx]):
                shape[idx] = shapeb[idx]
        return builtins.tensor(typea.T[0], tuple(shape))

    def _shape_as_ints(self, shape):
        """Convert a list of dimensions to ints and symbols"""
        return [s if is_symbolic(s) else int(s) for s in shape]

    def all_inputs_have_values(self, node):
        return all(self.gdict[i].attr['symbolic_value'] is not None for i in node.inputs)

    def any_inputs_have_values(self, node):
        return any(self.gdict[i].attr['symbolic_value'] is not None for i in node.inputs)

    def all_inputs_have_non_sym_values(self, node):
        return all(
            self.gdict[i].attr['symbolic_value'] is not None
            and not any_symbolic_or_unknown(self.gdict[i].attr['symbolic_value'].val)
            for i in node.inputs)

    def get_all_input_values(self, node):
        ret = []
        for i in node.inputs:
            if self.gdict[i].attr['symbolic_value'] is not None:
                ret.append(self.gdict[i].attr['symbolic_value'].val)
            else:
                ret.append(None)
        return ret

    def resolve_to_non_sym_val_or_die(self, node_name):
        """
        Requires node_name to resolve to non-symbolic value.
        """
        self.visit(node_name)
        val = try_get_non_sym_val(self.gdict[node_name])
        assert val is not None, "%s has to have non-symbolic value" % node_name
        return val

    def _promoted_primitive_type(self, type1, type2):
        """
        Given a pair of tensor or primitive types, find the smallest type that can store an instance
        of their primitive type.
        """
        ptype1 = type1.get_primitive() if builtins.is_tensor(type1) else type1
        ptype2 = type2.get_primitive() if builtins.is_tensor(type2) else type2
        return promote_types(ptype1, ptype2)

    def _get_node(self, node):
        if isinstance(node, six.string_types):
            node = self.gdict.get(node, None)
        return node

    #
    # Symbolic evaluation
    #

    def _get_symbolic_value(self, node):
        return self._get_node(node).attr.get('symbolic_value', None)

    def _set_symbolic_value(self, node, datatype, value):
        v = datatype()
        v.val = value
        self._get_node(node).attr['symbolic_value'] = v

    def _eval_symbolic_value_map(self, node, value_type):
        """
        Set a node's symbolic value by applying a projection.

        See _SYMBOL_MAP_OPS for the defintion of op name to function.

        Args:
            node (ParsedNode): The node whose symbolic value to set.
            value_type (nitro_builtin): The symbol's type.
        """
        input0 = self._get_symbolic_value(node.inputs[0])
        input1 = self._get_symbolic_value(node.inputs[1])
        if input0 is None or input1 is None:
            return

        binary_op = _SYMBOL_MAP_OPS.get(node.op, None)
        if binary_op is None:
            logging.warning('Symbolic evaluation of operator %s not implemented', node.op)
            return

        input0 = input0.val
        input1 = input1.val
        value = binary_op(input0, input1)
        self._set_symbolic_value(node, value_type, value)

    def _eval_symbolic_value_reduce(self, node, value_type):
        """
        Set a node's symbolic value by applying a reduction.

        See _SYMBOL_REDUCE_OPS for the defintion of op name to function.

        Args:
            node (ParsedNode): The node whose symbolic value to set.
            value_type (nitro_builtin): The symbol's type.
        """
        input0 = self._get_symbolic_value(node.inputs[0])
        input1 = self._get_symbolic_value(node.inputs[1])
        if input0 is None or input1 is None:
            return

        reduce_op = _SYMBOL_REDUCE_OPS.get(node.op, None)
        if reduce_op is None:
            logging.warning('Symbolic evaluation of operator %s not implemented', node.op)
            return

        values = input0.val
        axis = input1.val
        if not np.isscalar(axis) and len(axis) == 1:
            axis = axis[0]

        val = reduce_op(values, axis=axis)
        if builtins.is_tensor(value_type) and np.isscalar(val):
            val = np.array([val])

        self._set_symbolic_value(node, value_type, val)

    #
    # Common patterns
    #

    def _visit_unary(self, node):
        if len(node.inputs) != 1:
            raise ValueError('Expected 1 input to {} node {}'.format(node.op, node.name))
        return self.visit(node.inputs[0])

    def _visit_reduce(self, node):
        if len(node.inputs) != 2:
            raise ValueError('Expected 2 inputs to {} node {}'.format(node.op, node.name))
        typea = self.visit(node.inputs[0])
        typeb = self.visit(node.inputs[1])
        reduction_indices = self.gdict[node.inputs[1]].attr['symbolic_value']
        if typea is None:
            return None
        if reduction_indices is None:
            raise TypeError(
                "Cannot infer shape of {} because we cannot infer the value of reduction_indices".
                format(node.op))
        reduction_indices = reduction_indices.val
        # the reduction_idx node can be a scalar
        if not builtins.is_tensor(typeb):
            reduction_indices = [reduction_indices]
        keepdims = node.attr.get('keep_dims', False)
        reduced_shape = list(typea.get_shape())
        if len(reduction_indices) == 0:
            reduction_indices = list(range(len(reduced_shape)))
        if keepdims:
            for i in reduction_indices:
                reduced_shape[i] = 1
        else:
            # sort reverse so we can delete shape elements it back to front
            reduction_indices = sorted(reduction_indices)[::-1]
            for i in reduction_indices:
                reduced_shape.pop(i)
        if len(reduced_shape) == 0:
            rettype = typea.get_primitive()
        else:
            rettype = builtins.tensor(typea.get_primitive(), reduced_shape)
        node.attr['reduction_indices'] = reduction_indices
        node.attr['keep_dims'] = keepdims

        self._eval_symbolic_value_reduce(node, rettype)

        return rettype

    def _broadcast_shape(self, node, shapea, shapeb):
        """
        Determine the shape of a broadcast of two shapes.

        Args:
            node (ParsedNode): The node bring processed (used for exception messages).
            shapea (Iterable[int]): A shape
            shapeb (Iterable[int]): Another shape
        """
        shapea = list(shapea)
        shapeb = list(shapeb)
        if len(shapea) < len(shapeb):
            shapea = ([1] * (len(shapeb) - len(shapea))) + shapea
        if len(shapeb) < len(shapea):
            shapeb = ([1] * (len(shapea) - len(shapeb))) + shapeb
        # get loosest shape
        retshape = []
        for i in range(len(shapea)):
            a_unknown = is_symbolic_or_unknown(shapea[i])
            b_unknown = is_symbolic_or_unknown(shapeb[i])
            if shapea[i] == 1:
                retshape.append(shapeb[i])
            elif shapeb[i] == 1:
                retshape.append(shapea[i])
            elif not b_unknown and shapeb[i] > 1:
                if not a_unknown and shapea[i] != shapeb[i]:
                    raise ValueError(
                        'Incompatible dimension {} in {} operation {}'.format(
                            i, node.op, node.name))
                retshape.append(shapeb[i])
            elif not a_unknown and shapea[i] > 1:
                if not b_unknown and shapea[i] != shapeb[i]:
                    raise ValueError(
                        'Incompatible dimension {} in {} operation {}'.format(
                            i, node.op, node.name))
                retshape.append(shapea[i])
            elif a_unknown or b_unknown:
                retshape.append(sm.functions.Max(shapea[i], shapeb[i]))
            else:
                assert (shapea[i] == shapeb[i])
                retshape.append(shapea[i])
        return retshape

    def _visit_broadcast(self, node, is_predicate=False):
        # this is broadcast mul
        assert (len(node.inputs) == 2)
        typea = self.visit(node.inputs[0])
        typeb = self.visit(node.inputs[1])
        if typea is not None and typeb is not None:
            primitive_type = builtins.bool if is_predicate else self._promoted_primitive_type(
                typea, typeb)
            if primitive_type is None:
                raise ValueError('Incompatible primitive types in broadcast operation')
            if builtins.is_tensor(typea):
                if builtins.is_tensor(typeb):
                    retshape = self._broadcast_shape(node, typea.get_shape(), typeb.get_shape())
                    retval = builtins.tensor(primitive_type, retshape)
                else:
                    # a is tensor, b is not
                    retval = builtins.tensor(primitive_type, typea.get_shape())
            elif builtins.is_tensor(typeb):
                # b is tensor, a is not
                retval = builtins.tensor(primitive_type, typeb.get_shape())
            else:
                # both typea and typeb are not tensors
                retval = primitive_type
            self._eval_symbolic_value_map(node, retval)
            return retval
        else:
            # we have no idea what a and b are. Maybe Tensorflow does.
            return self._get_type_from_attr(node)

    # The main visitors

    def visit_get_tuple(self, node):  # DO NOT PROPAGATE TYPE INFERENCE ACROSS FUNCTIONS
        assert (len(node.inputs) == 1)
        parent_type = self.visit(node.inputs[0])
        self.propagate_tensor_array(node)
        # parent_type should be an instance of tuple
        if parent_type is None:
            return None
        assert (builtins.is_tuple(parent_type))
        parent_val = self.gdict[node.inputs[0]].attr['symbolic_value']
        rettype = parent_type.T[node.attr["index"]]
        if parent_val is not None:
            node.attr['symbolic_value'] = rettype()
            node.attr['symbolic_value'].val = parent_val[node.attr['index']].val

        return rettype

    def visit_Identity(self, node):
        ret = self._visit_unary(node)
        node.attr['symbolic_value'] = self.gdict[node.inputs[0]].attr['symbolic_value']
        if 'tensorarray_source' in self.gdict[node.inputs[0]].attr:
            node.attr['tensorarray_source'] = self.gdict[node.inputs[0]].attr['tensorarray_source']
        return ret

    def visit_ZerosLike(self, node):
        return self._visit_unary(node)

    def visit_Print(self, node):
        # this is just identity
        node.op = 'Identity'
        return self.visit(node.inputs[0])

    def visit_Log(self, node):
        ret = self._visit_unary(node)
        return ret

    def visit_Add(self, node):
        return self._visit_broadcast(node)

    def visit_AddV2(self, node):
        return self._visit_broadcast(node)

    def visit_Maximum(self, node):
        return self._visit_broadcast(node)

    def visit_Minimum(self, node):
        return self._visit_broadcast(node)

    def visit_LogicalOr(self, node):
        return self._visit_broadcast(node)

    def visit_LogicalAnd(self, node):
        return self._visit_broadcast(node)

    def visit_LogicalNot(self, node):
        return self.visit(node.inputs[0])

    def visit_All(self, node):
        return self._visit_reduce(node)

    def visit_Any(self, node):
        return self._visit_reduce(node)

    def visit_ArgMax(self, node):
        return self._visit_reduce(node)

    def visit_ArgMin(self, node):
        return self._visit_reduce(node)

    def visit_Prod(self, node):
        return self._visit_reduce(node)

    def visit_Assign(self, node):
        assert (len(node.inputs) == 2)
        return self.visit(node.inputs[1])

    def visit_Assert(self, node):
        pass

    def visit_BiasAdd(self, node):
        return self._visit_broadcast(node)

    def visit_Cast(self, node):
        assert (len(node.inputs) == 1)
        input_type = self.visit(node.inputs[0])
        dst_type = node.attr.get('DstT', None)
        if not builtins.is_primitive(dst_type):
            raise ValueError('Invalid destination type for Cast operation')
        if builtins.is_tensor(input_type):
            rettype = builtins.tensor(dst_type, input_type.get_shape())
        else:
            rettype = dst_type

        value = self._get_symbolic_value(node.inputs[0])
        if value is not None and not any_symbolic_or_unknown(value.val):
            self._set_symbolic_value(node,rettype,value.val.astype(builtins.utils.nptype_from_builtin(dst_type)))
        return rettype

    def visit_Concat(self, node):
        return self.visit_ConcatV2(node, is_v2=False)

    def visit_ConcatV2(self, node, is_v2=True):
        # Concat takes two tensors and a "axis to be concated"
        # get most specific type of all the concated variables
        def axis_dim_len(input_types, concat_axis):
            """Compute the length of the axis dimension"""
            new_axis_shape = 0
            for t in input_types:
                if builtins.is_tensor(t):
                    if len(t.get_shape()) > concat_axis:
                        taxis = t.get_shape()[concat_axis]
                        if taxis == -1:
                            new_axis_shape = make_symbol(node.name + '_new_axis')
                            break
                        else:
                            new_axis_shape += taxis
                    else:
                        new_axis_shape = make_symbol(node.name + '_new_axis')
                        break
                else:
                    new_axis_shape = make_symbol(node.name + '_new_axis')
                    break
            return new_axis_shape

        if len(node.inputs) < 2:
            raise ValueError('Expected at least 2 inputs to {} node {}'.format(node.op, node.name))

        # Axis arg must be a scalar
        if is_v2:
            axis_node = node.inputs[-1]
        else:
            axis_node = node.inputs[0]

        axis_type = self.visit(axis_node)

        if not builtins.is_primitive(axis_type):
            raise ValueError(
                'Unexpected non-primitive axis argument to {} op {}'.format(node.op, node.name))

        # Non-axis args must be tensors
        input_names = [inp for inp in node.inputs if inp != axis_node]
        input_types = [self.visit(inp) for inp in input_names]

        # If not able to infer type for any one of input, return None
        if None in input_types:
            return None

        if not all([builtins.is_tensor(it) for it in input_types]):
            raise ValueError(
                'Unexpected non-tensor argument to {} op {}'.format(node.op, node.name))
        rank = len(input_types[0].get_shape())

        # Axis must be computable at compile time
        concat_axis = self._get_symbolic_value(axis_node)
        if concat_axis is None:
            return None
        concat_axis = int(concat_axis.val)

        if concat_axis < 0:
            concat_axis += rank

        if concat_axis >= rank:
            raise ValueError('Axis out of bounds in {} op {}'.format(node.op, node.name))

        # Output shape has same rank as inputs and same size of
        # all non-axis dimensions
        if any([len(it.get_shape()) != rank for it in input_types[1:]]):
            raise ValueError('Inputs to {} op {} are not of same rank'.format(node.op, node.name))

        # Validate primitive types match and non-axis dimensions match
        retshape = list(input_types[0].get_shape())
        retprim = input_types[0].get_primitive()
        for it in input_types[1:]:
            if self.pedantic and it.get_primitive() != retprim:
                raise ValueError('Primitive type mismatch in {} op {}'.format(node.op, node.name))
            it_shape = it.get_shape()
            for i in range(rank):
                if i != concat_axis and retshape[i] != it_shape[i]:
                    if is_symbolic_or_unknown(retshape[i]) or is_symbolic_or_unknown(it_shape[i]):
                        continue
                    raise ValueError('Dimension mismatch in {} op {}'.format(node.op, node.name))

        retshape[concat_axis] = axis_dim_len(input_types, concat_axis)
        rettype = builtins.tensor(retprim, retshape)

        # Construct symbolic_value only if the inputs without
        # symbolic_value has 1 entry.
        create_symbolic = True
        for t, n in zip(input_types, input_names):
            if self._get_symbolic_value(n) is None and \
                builtins.is_tensor(t) and t.get_shape() != (1,):
                create_symbolic = False
                break
        if create_symbolic:
            inputs = self.get_all_input_values(node)
            inputs = inputs[:-1] if is_v2 else inputs[1:]
            for i in range(len(inputs)):
                if inputs[i] is None:
                    if builtins.is_tensor(input_types[i]):
                        # input_types[i] must of shape [1,]
                        inputs[i] = np.array([make_symbol(node.name + '%d' % i)])
                    else:
                        inputs[i] = make_symbol(node.name + '_%d' % i)
                if isscalar(inputs[i]):
                    inputs[i] = np.array(inputs[i])
            val = np.concatenate(inputs, axis=concat_axis)
            self._set_symbolic_value(node, rettype, val)
        return rettype

    def visit_Const(self, node):
        assert (len(node.inputs) == 0)
        node.attr['symbolic_value'] = node.value
        if node.datatype is not None:
            return node.datatype
        return self._get_type_from_attr(node)

    def _conv2d_strides_or_dilations(self, name, value, data_format, default_value):
        if value is None:
            value = default_value
        if not isinstance(value, (int, list)):
            raise ValueError('{} must be an int or list'.format(name))

        if isinstance(value, int):
            return [value] * 2

        if len(value) == 1:
            return value * 2
        if len(value) == 2:
            return value
        if len(value) != 4:
            raise ValueError('{} must have length 1, 2, or 4'.format(name))

        if data_format == "NHWC":
            # Only support stride/dilation along N, C == 1
            if not (value[0] == value[3] == 1):
                raise ValueError('{} along N and C other than 1 not implemented'.format(name))
            return value[1:3]

        # "NCHW"
        if not (value[0] == value[1] == 1):
            raise ValueError('{} along N and C other than 1 not implemented'.format(name))
        return value[2:]

    def _conv2d_pad(self, algorithm, custom_pad, filter_hw):
        # pad = [t+b, l+r]
        if algorithm == 'VALID':
            return [0] * 2
        if algorithm == 'SAME':
            return [(f // 2) * 2 for f in filter_hw]
        if algorithm == 'CUSTOM':
            if not isinstance(custom_pad, list) or len(custom_pad) != 4:
                raise ValueError('Invalid custom padding; expected (t, b, l, r)')
            return [custom_pad[0] + custom_pad[1], custom_pad[2] + custom_pad[3]]
        raise ValueError('Invalid padding algorithm "{}"'.format(algorithm))

    def visit_Conv2D(self, node):
        """
        Inputs:
            0 (str): The name of a 4D tensor in the format indicated by 'data_format' attribute
            1 (str): The name of a 4D filter of shape [height, width, in_channels, out_channels]
        Attributes:
            data_format (str): 'NWHC' or 'NCHW'
            strides (list): Sliding window stride: len(strides) is in (1, 2, 4)
            padding (str): 'SAME', 'VALID', or 'CUSTOM'. If 'CUSTOM', attribute 'pad' must also
                be specified.
            pad (list): Non-negative ints in order (t, b, l, r) indicating padding along H and W.
            dilations (list): OPTIONAL list of ints of length 1, 2, or 4 indicating dilation
                factor for each dimension.
        """
        if len(node.inputs) != 2:
            raise ValueError('Expected 2 inputs to {} node {}'.format(node.op, node.name))

        input_type = self.visit(node.inputs[0])
        filter_type = self.visit(node.inputs[1])

        for idx, input in enumerate(node.inputs):
            node_type = self._get_node(input).datatype
            if not builtins.is_tensor(node_type) or len(node_type.get_shape()) != 4:
                raise ValueError(
                    'Input {} to {} node {} is not a 4D tensor'.format(idx, node.op, node.name))
            for s in node_type.get_shape():
                if not isinstance(s, (six.integer_types, np.generic, sm.Basic)):
                    raise ValueError(
                        'Input and filter shapes must be int or symbolic in {} node {}'.format(
                            node.op, node.name))

        in_format = node.attr.get('data_format')
        if in_format not in ["NHWC", "NCHW"]:
            raise ValueError(
                'Invalid data_format "{}" in {} node {}'.format(in_format, node.op, node.name))

        padding = node.attr.get('padding')
        if padding not in ['VALID', 'SAME', 'CUSTOM']:
            raise ValueError(
                'Invalid padding algorithm "{}" in {} node {}'.format(padding, node.op, node.name))

        inshape = input_type.get_shape()
        filtshape = filter_type.get_shape()

        HW_strides = self._conv2d_strides_or_dilations(
            'strides', node.attr.get('strides'), in_format, 1)
        HW_dilations = self._conv2d_strides_or_dilations(
            'dilations', node.attr.get('dilations'), in_format, 1)
        pad = self._conv2d_pad(
            node.attr.get('padding'), node.attr.get('pad'),
            filtshape[:2])  # filtshape[:2] is kH, kW

        # TODO(daviddai): Dilation can be handled analogously as strides, but
        # SSA shouldn't overfit to TF's terrible strides / dilations specs (we
        # should disallow strides/dilation along N, C dimension). We should
        # also allow spatial dimensions beyond 2D
        #if not all(d == 1 for d in HW_dilations):
        #    raise NotImplementedError('Dilations other than 1 not implemented')

        N = inshape[0]
        C = inshape[3] if in_format == "NHWC" else inshape[1]  # "NCHW"
        HW_in = inshape[1:3] if in_format == "NHWC" else inshape[2:]  # "NCHW"
        filtershape = [HW_dilations[r] * (filtshape[r] - 1) + 1 for r in range(2)]
        HW_out_shape = [
            (HW_in[r] + pad[r] - filtershape[r]) // HW_strides[r] + 1
            for r in range(2)
        ]
        HW_out_shape = self._shape_as_ints(HW_out_shape)

        if node.op == 'DepthwiseConv2dNative':
            out_channels = filtshape[2] * filtshape[3]
        else:
            out_channels = filtshape[3]

        if in_format.startswith('NH'):
            retshape = [N] + HW_out_shape + [out_channels]
        else:
            retshape = [N, out_channels] + HW_out_shape
        return builtins.tensor(input_type.get_primitive(), tuple(retshape))


    def visit_DepthwiseConv2dNative(self, node):
        return self.visit_Conv2D(node)

    def visit_Conv2DBackpropInput(self, node):
        attr_output_type = self._get_type_from_attr(node)

        if attr_output_type is not None:
            return attr_output_type
        else:
            raise ValueError("[Type Inference] Conv2DBackpropInput type "
                             "inference case not handled")

    def visit_ResizeBilinear(self, node):
        return self._get_type_from_attr(node)

    def visit_ResizeNearestNeighbor(self, node):
        return self._get_type_from_attr(node)

    def _get_window_shape(self, ksize, height_idx):
        if not isinstance(ksize, collections.Sized):
            ksize = [ksize]
        if len(ksize) == 1:
            return (ksize[0], ksize[0])
        elif len(ksize) == 2:
            return tuple(ksize)
        elif len(ksize) == 4:
            return list(ksize[height_idx:height_idx + 2])
        raise ValueError("Expected ksize to be scalar or length 1, 2, or 4")

    # The documentation for tf.nn.convolution has a good description
    # of the proper output size.
    # https://www.tensorflow.org/api_docs/python/tf/nn/convolution
    def _get_window_reduced_size(self, algorithm, input_size, window_size, stride):
        if algorithm == 'VALID':
            return sm.ceiling((input_size - (window_size - 1)) / stride)
        if algorithm == 'SAME':
            return sm.ceiling((input_size / stride))
        raise ValueError(
            'Invalid padding algorithm "{}"; expected "SAME" or "VALID"'.format(algorithm))

    def _visit_pooling(self, node):
        if len(node.inputs) != 1:
            raise ValueError('Expected 1 inputs to {} node {}'.format(node.op, node.name))
        input_type = self.visit(node.inputs[0])
        if input_type is None:
            return self._get_type_from_attr(node)

        data_format = node.attr.get('data_format')
        if data_format not in ["NHWC", "NCHW"]:
            raise ValueError(
                'Invalid data_format "{}" in {} node {}'.format(data_format, node.op, node.name))

        padding = node.attr.get('padding')
        if padding not in ['VALID', 'SAME']:
            raise ValueError(
                'Invalid padding algorithm "{}" in {} node {}'.format(padding, node.op, node.name))

        strides = self._conv2d_strides_or_dilations(
            'strides', node.attr.get('strides'), data_format, 1)

        height_idx = 1 if data_format.startswith('NH') else 2  # NHWC, NCHW, or NCHW_VECT_C
        ksize = node.attr.get('ksize', [1])
        (window_height, window_width) = self._get_window_shape(ksize, height_idx)

        inshape = input_type.get_shape()
        filtshape = list(inshape)
        filtshape[height_idx] = self._get_window_reduced_size(
            padding, inshape[height_idx], window_height, strides[0])
        filtshape[height_idx + 1] = self._get_window_reduced_size(
            padding, inshape[height_idx + 1], window_width, strides[1])
        return builtins.tensor(input_type.get_primitive(), tuple(filtshape))

    def visit_MaxPool(self, node):
        return self._visit_pooling(node)

    def visit_AvgPool(self, node):
        return self._visit_pooling(node)

    def visit_Equal(self, node):
        return self._visit_broadcast(node, is_predicate=True)

    def visit_NotEqual(self, node):
        return self._visit_broadcast(node, is_predicate=True)

    def visit_ExpandDims(self, node):
        """
        Inputs:
            0 (str): The name of a tensor or scalar.
            1 (str): The name of an int indicating the dimension index to expand. Must be in
                     range [-rank(input) - 1, rank(input)] and able to be determined at compile
                     time.
        """
        if len(node.inputs) != 2:
            raise ValueError('Expected 2 inputs to {} node {}'.format(node.op, node.name))

        typea = self.visit(node.inputs[0])
        if not builtins.is_tensor(typea):
            typea = builtins.tensor(typea, (1, ))
            shape = []
        else:
            shape = list(typea.get_shape())  # input[0] should be a tensor.

        axis_type = self.visit(node.inputs[1])
        axis_value = None
        if builtins.is_tensor(axis_type):
            axis_shape = axis_type.get_shape()
            size = 1
            for s in axis_shape:
                size *= s
            if size != 1:
                raise ValueError(
                    'Unexpected value for axis specified for {} node {}'.format(node.op, node.name))
            axis_value = self._get_symbolic_value(node.inputs[1]).val[0]
        elif builtins.is_int(axis_type):
            axis_value = self._get_symbolic_value(node.inputs[1]).val
        else:
            raise ValueError(
                'Unexpected non-int axis specified for {} node {}'.format(node.op, node.name))

        if axis_value < -len(typea.get_shape()) - 1 or axis_value > len(typea.get_shape()):
            raise IndexError(
                'Axis value {} is out of bounds for {} node {}'.format(
                    axis_value, node.op, node.name))

        if axis_value < 0:
            cut = len(shape) + axis_value + 1
        else:
            cut = axis_value
        shape = shape[0:cut] + [1] + shape[cut:]

        rettype = builtins.tensor(typea.get_primitive(), tuple(shape))
        input_val = self._get_symbolic_value(node.inputs[0])
        if input_val is not None:
            input_val = np.array(input_val.val).reshape(shape)
            self._set_symbolic_value(node, rettype, input_val)
        return rettype

    def visit_Fill(self, node):
        """
        Inputs:
            0 (str): The name of a tensor describing the size of tensor to create.
            1 (str): The name of a scalar value to fill the new tensor with.
        """
        if len(node.inputs) != 2:
            raise ValueError('Expected 2 inputs to {} node {}'.format(node.op, node.name))

        typea = self.visit(node.inputs[0])
        typeb = self.visit(node.inputs[1])

        shape_value = self._get_symbolic_value(node.inputs[0])
        if shape_value is not None:
            shape_value = shape_value.val.flatten()
            shape = tuple([int(s) if not is_symbolic(s) else s for s in shape_value])
            rettype = builtins.tensor(typeb, shape)

            fill_value = self._get_symbolic_value(node.inputs[1])
            if fill_value is not None and not any_symbolic_or_unknown(shape):
                value = np.ones(shape, dtype=builtins.utils.nptype_from_builtin(typeb)) * fill_value.val
                self._set_symbolic_value(node, rettype, value)
        else:
            # shape unknown.
            # we should be able to derive a rank
            shape = tuple(make_symbol(node.name + str(i)) for i in range(typea.get_shape()[0]))
            rettype = builtins.tensor(typeb, shape)
        return rettype

    def visit_RandomUniform(self, node):
        """
        Input:
            0 (str): The name of a 1-D tensor indicating output shape
        Attributes:
            dtype (builtin): The scalar type to generate
        """
        if len(node.inputs) != 1:
            raise ValueError('Expected 1 input to {} node {}'.format(node.op, node.name))

        # input[0] is the shape
        # the value that would be in the tensor(shape=input[0])
        shape_type = self.visit(node.inputs[0])
        if not builtins.is_tensor(shape_type) or len(shape_type.get_shape()) != 1:
            raise ValueError('Input must be a 1-D tensor to {} node {}'.format(node.op, node.name))

        dtype = node.attr.get('dtype')
        if dtype is None:
            raise ValueError(
                'dtype is a required attribute in {} node {}'.format(node.op, node.name))
        if not builtins.is_scalar(dtype):
            raise ValueError('dtype must be a scalar type in {} node {}'.format(node.op, node.name))

        shape_value = self._get_symbolic_value(node.inputs[0])
        if shape_value is not None:
            shape = tuple(shape_value.val.flatten())
            rettype = builtins.tensor(dtype, shape)
            return rettype

        # shape unknown.
        # I should be able to derive a rank
        shape = tuple(make_symbol(node.name + str(i)) for i in range(len(shape_type.get_shape())))
        rettype = builtins.tensor(dtype, shape)
        return rettype

    def visit_FloorMod(self, node):
        return self._visit_broadcast(node)

    def visit_Pow(self, node):
        return self._visit_broadcast(node)

    def visit_function(self, node):
        pass

    def visit_function_entry(self, node):
        pass

    def visit_Gather(self, node):
        params_type = self.visit(node.inputs[0])
        indices_type = self.visit(node.inputs[1])
        axis_value = 0
        if len(node.inputs) == 3:
            axis = self.visit(node.inputs[2])
            axis_value = self.gdict[node.inputs[2]].attr['symbolic_value'].val
        node.attr['axis'] = axis_value
        if params_type is None or indices_type is None:
            return None
        if not builtins.is_tensor(indices_type):
            indices_shape = []
        else:
            indices_shape = list(indices_type.get_shape())
        params_shape = list(params_type.get_shape())
        retshape = params_shape[:axis_value] + indices_shape + params_shape[axis_value + 1:]
        if len(indices_shape) == 0 and len(params_shape) == 1:
            # For scalar indices, rank(output) == rank(params) - 1 is the only
            # possibility for gather to return non-tensor.
            rettype = params_type.get_primitive()
        else:
            rettype = builtins.tensor(params_type.get_primitive(), retshape)

        if self.gdict[node.inputs[0]].attr['symbolic_value'] is not None and \
                self.gdict[node.inputs[1]].attr['symbolic_value'] is not None and \
                axis_value is not None:
            params_val = self.gdict[node.inputs[0]].attr['symbolic_value'].val
            indices_val = self.gdict[node.inputs[1]].attr['symbolic_value'].val
            retval = np.take(params_val, indices_val, axis=axis_value)
            retval = try_to_np_type(retval)
            node.attr['symbolic_value'] = rettype()
            node.attr['symbolic_value'].val = retval
        return rettype

    def visit_GatherV2(self, node):
        node.op = 'Gather'
        return self.visit_Gather(node)

    def visit_GatherNd(self, node):
        params_type = self.visit(node.inputs[0])
        indices_type = self.visit(node.inputs[1])
        if params_type is None or indices_type is None:
            return None

        indices_shape = []
        if not builtins.is_tensor(indices_type):
            indices_shape = []
        else:
            indices_shape = list(indices_type.get_shape())
        params_shape = list(params_type.get_shape())
        retshape = indices_shape[:-1] + params_shape[indices_shape[-1]:]
        rettype = builtins.tensor(params_type.get_primitive(), retshape)

        return rettype

    def visit_ScatterNd(self, node):
        indices_type = self.visit(node.inputs[0])
        updates_type = self.visit(node.inputs[1])
        shapes_type = self.visit(node.inputs[2])
        if updates_type is None or shapes_type is None:
            return None

        retshape = []
        if 'symbolic_value' in self.gdict[node.inputs[2]].attr:
            size = list(self.gdict[node.inputs[2]].attr['symbolic_value'].val)
            for i in range(len(size)):
                if is_symbolic_or_unknown(size[i]):
                    retshape.append(make_symbol(node.name + '_' + str(i)))
                else:
                    retshape.append(size[i])
            if len(retshape) == 0:
                rettype = updates_type.get_primitive()
            else:
                rettype = builtins.tensor(updates_type.get_primitive(), retshape)

        rettype = builtins.tensor(updates_type.get_primitive(), retshape)

        return rettype

    def visit_GatherTree(self, node):
        # TODO: To implement?
        return self._get_type_from_attr(node)

    def visit_GreaterEqual(self, node):
        return self._visit_broadcast(node, is_predicate=True)

    def visit_Greater(self, node):
        return self._visit_broadcast(node, is_predicate=True)

    def visit_Less(self, node):
        return self._visit_broadcast(node, is_predicate=True)

    def visit_LessEqual(self, node):
        return self._visit_broadcast(node, is_predicate=True)

    def visit_make_tuple(self, node):
        types = [self.visit(i) for i in node.inputs]
        self.propagate_tensor_array(node)

        if any([t is None for t in types]):
            logging.warning("make_tuple at %s has an unknown type %s", node.name, str(types))
        types = [t if t is not None else builtins.unknown for t in types]
        return builtins.tuple(types)

    def visit_BatchMatMul(self, node):
        # node.op = "MatMul"
        # Batch matmul was deprecated, we implement this in MatMul
        return self.visit_MatMul(node)

    def visit_BatchMatMulV2(self, node):
        node.op = 'BatchMatMul'
        return self.visit_BatchMatMul(node)

    def _shape_transposed(self, shape):
        shape = list(shape)
        shape[-1], shape[-2] = shape[-2], shape[-1]
        return tuple(shape)

    def visit_MatMul(self, node):
        """
        Inputs:
            0 (str): Name of a tensor with rank >= 2 after any transpositions
            1 (str): Name of a tensor with rank >= 2 after any transpositions
        Attributes:
            transpose_a (bool): If True, transpose the first input before multiplying
            transpose_b (bool): If True, transpose the second input before multiplying
            adj_x (bool): If True, adjoint the first input before multiplying
            adj_y (bool): If True, adjoint the second input before multiplying
        """
        #
        # Validate inputs
        #
        if len(node.inputs) != 2:
            raise ValueError('Expected 2 inputs to {} node {}'.format(node.op, node.name))

        typea = self.visit(node.inputs[0])
        typeb = self.visit(node.inputs[1])
        if typea is None or typeb is None:
            return self._get_type_from_attr(node)

        if not builtins.is_tensor(typea) or not builtins.is_tensor(typeb):
            raise ValueError('Inputs must be tensors in {} node {}'.format(node.op, node.name))

        mata_shape = typea.get_shape()
        matb_shape = typeb.get_shape()
        if len(mata_shape) < 2 or len(matb_shape) < 2:
            raise ValueError(
                'Inputs must have rank 2 or greater in {} node {}'.format(node.op, node.name))

        #
        # Validate attributes
        # this handles the parameters from both MatMul and BatchMatMul
        #
        transpose_a = node.attr.get('transpose_a', False)
        transpose_b = node.attr.get('transpose_b', False)
        adj_x = node.attr.get('adj_x', False)
        adj_y = node.attr.get('adj_y', False)
        if (transpose_a and adj_x) or (transpose_b and adj_y):
            raise ValueError('transpose and adjoint are mutually exclusive a given input')
        transpose_a = adj_x or transpose_a
        transpose_b = adj_y or transpose_b

        # Apply transpositions
        if transpose_a:
            mata_shape = self._shape_transposed(mata_shape)
        if transpose_b:
            matb_shape = self._shape_transposed(matb_shape)

        # Check shape compatibility
        if not all(is_symbolic_or_unknown(s) for s in [mata_shape[-1], matb_shape[-2]]):
            if mata_shape[-1] != matb_shape[-2]:
                raise ValueError('Incompatible dimensions in {} op {}'.format(node.op, node.name))

        # Figure out the resulting shape. Outer dimensions are broadcastable.
        outera = mata_shape[0:-2]
        outerb = matb_shape[0:-2]
        outer_shape = self._broadcast_shape(node, outera, outerb)
        shape = outer_shape + [mata_shape[-2], matb_shape[-1]]

        if len(shape) > 2:
            node.op = 'BatchMatMul'

        primitive = self._promoted_primitive_type(typea, typeb)
        return builtins.tensor(primitive, tuple(shape))

    def visit_LSTMBlock(self, node):
        intype = self.visit(node.inputs[0])
        W_type = self.visit(node.inputs[1])

        mode = node.attr["mode"]
        shape = list(intype.get_shape())
        W_shape = list(W_type.get_shape())
        hidden_shape = W_shape[-1] / 4
        if node.attr.get("bidirectional", False):
            hidden_shape /= 2
        input_shape = W_shape[0] - hidden_shape
        assert shape[-1] == input_shape, "Input size doesn't match"
        shape[-1] = hidden_shape
        if mode == "cell":
            # returns output/cell state/hidden state
            types = [builtins.tensor(intype.get_primitive(), tuple(shape)) for _ in range(3)]
        elif mode == "encoder":
            hidden_shape = shape[:]
            output_shape = shape[:]
            if not node.attr["output_all_states"]:
                if node.attr["time_major"]:
                    output_shape[0] = 1
                else:
                    output_shape[1] = 1

            if node.attr.get("bidirectional", False):
                output_shape[-1] *= 2
                output_type = builtins.tensor(intype.get_primitive(), tuple(output_shape))
                hidden_type = builtins.tensor(intype.get_primitive(), tuple(hidden_shape))
                types = [output_type] + [hidden_type] * 4
            else:
                output_type = builtins.tensor(intype.get_primitive(), tuple(output_shape))
                hidden_type = builtins.tensor(intype.get_primitive(), tuple(hidden_shape))
                types = [output_type] + [hidden_type] * 2
        else:
            raise ValueError("Unknown mode type for LSTMBlock")

        return builtins.tuple(types)

    def visit_Mul(self, node):
        return self._visit_broadcast(node)

    def visit_Neg(self, node):
        return self._visit_unary(node)

    def visit_NoOp(self, node):
        return builtins.void

    def visit_Pack(self, node):
        input_values = []
        intype = None
        rettype = None
        for i in node.inputs:
            intype = self.visit(i)
            input_values.append(self.gdict[i].attr['symbolic_value'])
        if all(i is not None for i in input_values):
            # we can force the value!
            for i in range(len(input_values)):
                input_values[i] = input_values[i].val
                input_values[i] = np.array(input_values[i])
            val = np.stack(arrays=input_values, axis=node.attr['axis'])
            primitive = intype
            if builtins.is_tensor(intype):
                primitive = intype.get_primitive()
            rettype = builtins.tensor(primitive, tuple(val.shape))
            node.attr['symbolic_value'] = rettype()
            node.attr['symbolic_value'].val = val
        if rettype is not None:
            return rettype
        else:
            output_shapes = node.attr['_output_shapes']
            if len(output_shapes[0]) > 0:
                return builtins.tensor(node.attr['T'], tuple(output_shapes[0]))
            elif 'N' in node.attr:
                return builtins.tensor(node.attr['T'], (node.attr['N'], ))
        return None

    def visit_Pad(self, node):
        lefttype = self.visit(node.inputs[0])
        self.visit(node.inputs[1])
        s = self.gdict[node.inputs[1]].attr['symbolic_value']
        if not s:
            attr_type = self._get_type_from_attr(node)
            if not attr_type and self.gdict[node.inputs[1]].datatype and not any_symbolic_or_unknown(
                self.gdict[node.inputs[1]].datatype.T[1]):
                # at least we can get a rank
                rank = self.gdict[node.inputs[1]].datatype.T[1][0]
                ret_shape = [make_symbol(node.name + "_" + str(i)) for i in range(rank)]
                return builtins.tensor(lefttype.get_primitive(), ret_shape)
            else:
                return attr_type
        s = s.val
        assert len(s.shape) == 2, "padding specs must be of shape [r, 2]" \
            + "where r is rank of input tensor"
        if not builtins.is_tensor(lefttype):
            raise RuntimeError("Pad only operates on tensor type, but got " + str(lefttype))
        retshape = list(lefttype.get_shape())
        for i in range(len(retshape)):
            retshape[i] = retshape[i] + s[i][0] + s[i][1]
        rettype = builtins.tensor(lefttype.get_primitive(), retshape)
        left_sym_val = self.gdict[node.inputs[0]].attr["symbolic_value"]
        if left_sym_val:
            node.attr["symbolic_value"] = rettype()
            node.attr["symbolic_value"].val = np.pad(
                left_sym_val.val, s, "constant", constant_values=node.attr['constant_values'])
        return rettype

    def visit_PadV2(self, node):
        return self.visit_Pad(node)

    def visit_MirrorPad(self, node):
        return self.visit_Pad(node)

    def visit_Placeholder(self, node):
        return self._get_type_from_attr(node)

    def visit_PlaceholderWithDefault(self, node):
        return self._get_type_from_attr(node)

    def visit_Range(self, node):
        """
        Inputs:
            All must be int32, int64, float32, float64 or a single-value tensor thereof.

            If len(inputs) == 2:
                0: limit
                1: delta
            elif len(inputs) == 3:
                0: start
                1: limit
                2: delta
        """
        if len(node.inputs) not in (2, 3):
            raise ValueError('Expected 2 or 3 inputs to {} node {}'.format(node.op, node.name))

        # Ensure all inputs have valid types
        input_types = [self.visit(input) for input in node.inputs]
        if any([input_type is None for input_type in input_types]):
            # Non-const propagation.
            return None

        # Figure out the primitive return type
        # We use the highest-ranked type among our inputs
        dtypes = [builtins.int32, builtins.int64, builtins.fp32, builtins.fp64]
        dtype_ranks = dict(zip(dtypes, range(0, len(dtypes))))

        datatype = dtypes[0]
        for dt in input_types:
            if builtins.is_tensor(dt):
                dt_shape = dt.get_shape()
                if dt_shape and (len(dt_shape) != 1 or dt_shape[0] != 1):
                    raise ValueError(
                        'Invalid input tensor input with more than value in {} node {}'.format(
                            node.op, node.name))
                dt = dt.get_primitive()
            dt_rank = dtype_ranks.get(dt)
            if dt_rank is None:
                raise ValueError('Invalid input datatype to {} node {}'.format(node.op, node.name))
            if dt_rank > dtype_ranks[datatype]:
                datatype = dt

        # Ensure all inputs have symbolic values
        input_values = [self._get_symbolic_value(input) for input in node.inputs]
        if any([input_value is None for input_value in input_values]):
            # Non-fixed value propagation (e.g. TensorArray)
            return builtins.tensor(datatype, [make_symbol(node.name + '_range')])

        # Extract parameters from symbolic values
        input_values = [
            iv.val[0] if isinstance(iv.val, np.ndarray) else iv.val for iv in input_values
        ]

        # Interpret positional arguments
        if len(node.inputs) == 2:
            limit_type, delta_type = input_types
            start = 0
            limit, delta = input_values
        elif len(node.inputs) == 3:
            start_type, limit_type, delta_type = input_types
            start, limit, delta = input_values
        else:
            assert False, "logic error"

        # Figure out the node type
        shape = (limit - start) / delta
        shape = shape if is_symbolic(shape) else int(math.ceil(shape))
        rettype = builtins.tensor(datatype, [shape])

        # Evaluate the symbolic value
        if not any_symbolic_or_unknown([start, limit, delta]):
            nptype = builtins.utils.nptype_from_builtin(datatype)
            self._set_symbolic_value(
                node, rettype, np.arange(start=start, stop=limit, step=delta, dtype=nptype))
        elif delta == 1:
            self._set_symbolic_value(node, rettype, sm.Interval(start, limit))
        return rettype

    def visit_Rank(self, node):
        # This is also interesting. Tensorflow will return a 0-D tensor,
        # while we transformed 0-D tensor to scalar in parsing.
        input_type = self.visit(node.inputs[0])
        input_shape = input_type.get_shape()
        node.attr['symbolic_value'] = builtins.int32()
        node.attr['symbolic_value'].val = len(input_shape)
        return builtins.int32

    def visit_Relu(self, node):
        return self._visit_unary(node)

    def visit_PRelu(self, node):
        ret = self._visit_unary(node)

        # If alpha is specified as a node, see if we can get its value now.
        alpha_node = node.attr.get('alpha', None)
        if isinstance(alpha_node, six.string_types):
            alpha_type = self.visit(alpha_node)
            alpha = self._get_symbolic_value(alpha_node)
            if alpha is None:
                raise ValueError('PRelu alpha node could not be evaluated')
            alpha = alpha.val
            if isinstance(alpha, np.ndarray):
                if alpha.size != 1:
                    raise ValueError('PRelu alpha must be a single value')
                alpha = np.asscalar(alpha)
            node.attr['alpha'] = alpha
        return ret

    def visit_Relu6(self, node):
        return self._visit_unary(node)

    def visit_LeakyRelu(self, node):
        return self._visit_unary(node)

    def visit_Selu(self, node):
        return self._visit_unary(node)

    def visit_Reshape(self, node):
        def check_volumetric_constraint(left_volume, inshape):
            right_volume = 1
            left_symbols = set()
            right_symbols = set()
            try:
                left_symbols = left_volume.free_symbols
            except:
                pass
            try:
                right_symbols = right_volume.free_symbols
            except:
                pass
            # Generally, we want to solve for right in terms of left. But this
            # is kinda annoying actually.
            shape = list(inshape)
            for i in shape:
                right_volume = right_volume * i
            if is_symbolic(right_volume):
                constraints = [left_volume - right_volume]
                solve_for = [s for s in shape if is_symbolic(s)]

                for rightsym in solve_for:
                    sol = sm.solve(constraints, [rightsym], dict=True)
                    if not isinstance(sol, list):
                        sol = [sol]
                    # look for an acceptable solution
                    for s in sol:
                        if 0 in s.values():
                            continue
                        for i in range(len(shape)):
                            if shape[i] in s:
                                v = s[shape[i]]
                                if len(v.free_symbols - left_symbols) > 0:
                                    continue
                                try:
                                    shape[i] = int(v)
                                except:
                                    shape[i] = v
            return shape

        assert (len(node.inputs) == 2)
        lefttype = self.visit(node.inputs[0])
        if builtins.is_tensor(lefttype):
            left_primitive = lefttype.get_primitive()
            left_shape = lefttype.get_shape()
            left_volume = 1
            for i in left_shape:
                left_volume = left_volume * i
        else:
            left_primitive = lefttype
            left_volume = 1
        if lefttype is None:
            return None
        self.visit(node.inputs[1])
        if self.gdict[node.inputs[1]].attr['symbolic_value'] is not None:
            shape = list(self.gdict[node.inputs[1]].attr['symbolic_value'].val)
            replace_neg_1_with_symbolic(shape, node.name + '_shape')
            shape = check_volumetric_constraint(left_volume, shape)
            r = builtins.tensor(left_primitive, shape)
            if self.gdict[node.inputs[0]].attr['symbolic_value'] is not None \
                and all(isscalar(a) for a in shape):
                node.attr['symbolic_value'] = r()
                node.attr['symbolic_value'].val = reshape_with_symbol(
                    self.gdict[node.inputs[0]].attr['symbolic_value'].val, shape)
            return r

        # check if we have answer from attributes.
        # Otherwise the final fall back is just [-1] * rank
        try:
            attr_type = self._get_type_from_attr(node)
        except:
            attr_type = None
        if attr_type is not None:
            shape = check_volumetric_constraint(left_volume, attr_type.get_shape())
            return builtins.tensor(attr_type.get_primitive(), shape)
        elif self.gdict[node.inputs[1]].datatype is not None and not any_symbolic_or_unknown(
                self.gdict[node.inputs[1]].datatype.T[1]):
            # at least we can get a rank
            rank = self.gdict[node.inputs[1]].datatype.T[1][0]
            ret_shape = [make_symbol(node.name + "_" + str(i)) for i in range(rank)]
            return builtins.tensor(left_primitive, ret_shape)

    def visit_return(self, node):
        return self._visit_unary(node)

    def visit_ReverseSequence(self, node):
        assert (len(node.inputs) == 2)
        return self.visit(node.inputs[0])

    def visit_ReverseV2(self, node):
        assert (len(node.inputs) == 2)
        return self.visit(node.inputs[0])

    def visit_Sin(self, node):
        rettype = self._visit_unary(node)
        input = self.gdict[node.inputs[0]]
        if input.attr['symbolic_value'] is not None:
            node.attr['symbolic_value'] = rettype()
            node.attr['symbolic_value'].val = np.sin(input.attr['symbolic_value'].val)
        return rettype

    def visit_Cos(self, node):
        rettype = self._visit_unary(node)
        input = self.gdict[node.inputs[0]]
        if input.attr['symbolic_value'] is not None:
            node.attr['symbolic_value'] = rettype()
            node.attr['symbolic_value'].val = np.cos(input.attr['symbolic_value'].val)
        return rettype

    def visit_Tan(self, node):
        rettype = self._visit_unary(node)
        input = self.gdict[node.inputs[0]]
        if input.attr['symbolic_value'] is not None:
            node.attr['symbolic_value'] = rettype()
            node.attr['symbolic_value'].val = np.tan(input.attr['symbolic_value'].val)
        return rettype

    def visit_Tanh(self, node):
        rettype = self._visit_unary(node)
        input = self.gdict[node.inputs[0]]
        if input.attr['symbolic_value'] is not None:
            node.attr['symbolic_value'] = rettype()
            node.attr['symbolic_value'].val = np.tanh(input.attr['symbolic_value'].val)
        return rettype

    def visit_Sqrt(self, node):
        rettype = self._visit_unary(node)
        input = self.gdict[node.inputs[0]]
        if input.attr['symbolic_value'] is not None:
            node.attr['symbolic_value'] = rettype()
            node.attr['symbolic_value'].val = input.attr['symbolic_value'].val ** 0.5
        return rettype

    def visit_Rsqrt(self, node):
        rettype = self._visit_unary(node)
        input = self.gdict[node.inputs[0]]
        if input.attr['symbolic_value'] is not None:
            node.attr['symbolic_value'] = rettype()
            node.attr['symbolic_value'].val = input.attr['symbolic_value'].val ** -0.5
        return rettype

    def visit_Square(self, node):
        rettype = self._visit_unary(node)
        input = self.gdict[node.inputs[0]]
        if input.attr['symbolic_value'] is not None:
            node.attr['symbolic_value'] = rettype()
            node.attr['symbolic_value'].val = input.attr['symbolic_value'].val ** 2
        return rettype

    def visit_Exp(self, node):
        return self._visit_unary(node)

    def visit_Shape(self, node):
        # need to parse node itself.
        parent_type = self.visit(node.inputs[0])
        shape = []
        if parent_type is None or not builtins.is_tensor(parent_type):
            return builtins.tensor(builtins.int32, [make_symbol(node.name + '_shape')])
        if parent_type is not None:
            shape = parent_type.get_shape()
            rettype = builtins.tensor(builtins.int32, [len(shape)])
        else:
            rettype = builtins.tensor(builtins.int32, [make_symbol(node.name + '_shape')])
        if len(shape) > 0:
            # we have the true value
            node.attr['symbolic_value'] = rettype()
            node.attr['symbolic_value'].val = np.array(shape)
        return rettype

    def visit_Select(self, node):
        assert len(node.inputs) == 3
        typecond = self.visit(node.inputs[0])

        if builtins.is_tensor(typecond):
            # this is a masking op.
            # change the name
            node.op = 'SelectMask'

        typea = self.visit(node.inputs[1])
        typeb = self.visit(node.inputs[2])

        if all([builtins.is_tensor(atype) for atype in [typecond, typea, typeb]]):
            rankcond = len(typecond.get_shape())
            ranka = len(typea.get_shape())
            rankb = len(typeb.get_shape())

            assert (ranka == rankb)
            if rankcond == 1 and ranka > 1:
                node.attr['expand_dims'] = [-i - 1 for i in range(ranka - rankcond)]

        if typea is not None and typeb is not None:
            compatible, restype = builtins.is_tensor_and_is_compatible_general_shape(typea, typeb)
            if compatible:
                return restype
            elif typea == typeb:
                return typea
            else:
                logging.error(
                    "%s != %s", builtins.get_type_info(typea), builtins.get_type_info(typeb))

        if typea is not None:
            return typea
        else:
            return typeb

    def visit_SelectMask(self, node):
        return self.visit_Select(node)

    def visit_SelectV2(self, node):
        return self.visit_Select(node)

    def visit_iff(self, node):
        # an op we inserted. equivalent to the functional IF
        # IF cond: true: false
        assert (len(node.inputs) == 3)
        typecond = self.visit(node.inputs[0])
        # assert (builtins.is_tensor(typecond) == False)

        typea = self.visit(node.inputs[1])
        typeb = self.visit(node.inputs[2])
        if typea is not None and typeb is not None:

            compatible, restype = builtins.is_tensor_and_is_compatible_general_shape(typea, typeb)
            if compatible:
                return restype
            elif typea == typeb:
                return typea
            else:
                logging.warning(
                    "In an IFF node %s != %s", builtins.get_type_info(typea),
                    builtins.get_type_info(typeb))
                return typea

        if typea is not None:
            return typea
        else:
            return typeb

    def visit_Where(self, node):
        input_type = self.visit(node.inputs[0])
        if len(node.inputs) == 3 and builtins.is_tensor(input_type):
            return self.visit_Select(node)
        elif len(node.inputs) == 3:
            return self.visit_iff(node)
        else:
            assert (len(node.inputs) == 1)
            rank = len(self.gdict[node.inputs[0]].datatype.get_shape())
            ret_shape = [make_symbol(node.name + "_" + str(0)), rank]
            return builtins.tensor(builtins.int32, ret_shape)

    def visit_Sigmoid(self, node):
        return self._visit_unary(node)

    def visit_Elu(self, node):
        return self._visit_unary(node)

    def visit_Slice(self, node):
        for i in node.inputs:
            self.visit(i)
        input_type = self.visit(node.inputs[0])
        input_shape = input_type.get_shape()
        input_value = self.gdict[node.inputs[0]].attr['symbolic_value']
        try:
            begin = list(self.gdict[node.inputs[1]].attr['symbolic_value'].val)
            size = list(self.gdict[node.inputs[2]].attr['symbolic_value'].val)
            end = [
                int(begin[i] + size[i]) if size[i] != -1 else 2147483647 for i in range(len(begin))
            ]
            assert builtins.is_tensor(input_type)
            input_shape = input_type.get_shape()
            end = [min(i, j) for i, j in zip(end, input_shape)]
            size = [min(s, e - b) for s, b, e in zip(size, begin, end)]
            slices = [[int(begin[i]), int(end[i]), 1] for i in range(len(begin))]
            node.attr['slice'] = slices
            node.attr['begin_masks'] = [idx for idx, value in enumerate(begin) if value == 0]
            node.attr['end_masks'] = [idx for idx, value in enumerate(end) if value == 2147483647]
            node.attr['squeeze'] = []
            output_value = None
            if input_value is not None:
                slices = [slice(*i) for i in slices]
                slices = tuple(slices)
                res = input_value.val[slices]

                if isscalar(res):
                    rettype = input_type.get_primitive()
                    output_value = rettype
                    output_value.val = res
                elif not isscalar(res):
                    rettype = builtins.tensor(input_type.get_primitive(), res.shape)
                    output_value = rettype()
                    output_value.val = res
            else:
                retshape = []
                for i in range(len(begin)):
                    if is_symbolic_or_unknown(size[i]):
                        if is_symbolic_or_known(input_shape[i]) and is_symbolic_or_known(begin[i]):
                            retshape.append(input_shape[i] - begin[i])
                        else:
                            retshape.append(make_symbol(node.name + '_' + str(i)))
                    else:
                        retshape.append(size[i])
                if len(retshape) == 0:
                    rettype = input_type.get_primitive()
                else:
                    rettype = builtins.tensor(input_type.get_primitive(), retshape)
            node.attr['symbolic_value'] = output_value
        except:
            # unable to infer shape
            if 'slice' in node.attr:
                del node.attr['slice']
            node.attr['squeeze'] = []
            try:
                size = list(self.gdict[node.inputs[2]].attr['symbolic_value'].val)
                try:
                    begin = list(self.gdict[node.inputs[1]].attr['symbolic_value'].val)
                    begin = to_int(begin)
                    size = [
                        input_shape[i] - begin[i] if s in (-1, 2147483647) else s
                        for i, s in enumerate(size)
                    ]
                except:
                    # Adjust size if begin is available, otherwise trust the
                    # materialized size assuming it's reasonable.
                    if max(size) == 2147483647:
                        raise RuntimeError()
                size = to_int(size)

                if len(size) == 1 and size[0] == 1:
                    rettype = input_type.get_primitive()
                else:
                    rettype = builtins.tensor(input_type.get_primitive(), size)
                node.attr['generic_slice'] = True
                node.attr['size'] = size
            except:
                retshape = []
                for i in range(len(input_shape)):
                    retshape.append(make_symbol(node.name + '_' + str(i)))
                if len(retshape) == 0:
                    rettype = input_type.get_primitive()
                else:
                    rettype = builtins.tensor(input_type.get_primitive(), retshape)
        return rettype

    def visit_Softmax(self, node):
        return self._visit_unary(node)

    def visit_Softplus(self, node):
        return self._visit_unary(node)

    def visit_LogSoftmax(self, node):
        return self._visit_unary(node)

    def visit_Split(self, node, mode='Split'):
        datatype = None
        if 'T' in node.attr and node.attr['T'] is not None:
            datatype = node.attr['T']
        elif 'dtype' in node.attr and node.attr['dtype'] is not None:
            datatype = node.attr['dtype']
        # try to fill unknown output shapes from the input
        shapes = None
        num_split = None
        if 'num_split' in node.attr:
            num_split = node.attr['num_split']
        if '_output_shapes' in node.attr:
            shapes = node.attr['_output_shapes']
        split_dim_idx = 2 if mode == 'SplitV' else 0
        value_idx = 0 if mode == 'SplitV' else 1
        self.visit(node.inputs[split_dim_idx])
        if mode == 'SplitV':
            self.visit(node.inputs[1])
            if self.gdict[node.inputs[1]].attr['symbolic_value'] is not None:
                split_size_type = self.gdict[node.inputs[1]].datatype
                split_size = self.gdict[node.inputs[1]].attr['symbolic_value'].val
                if not builtins.is_tensor(split_size_type):
                    mode = 'Split'
                else:
                    num_split = split_size.shape[0]
                    node.attr['num_split'] = num_split
        try_materialize = False
        # this *must!* be constant
        if self.gdict[node.inputs[split_dim_idx]].attr['symbolic_value'] is not None:
            split_dim = self.gdict[node.inputs[split_dim_idx]].attr['symbolic_value'].val
            input_type = self.visit(node.inputs[value_idx])
            if datatype is None:
                datatype = input_type.get_primitive()
            node.attr['split_dim'] = int(split_dim)
            if input_type is not None:
                input_shape = input_type.get_shape()
                from_shapes_ok = False
                try:
                    if shapes is not None:
                        # use the type infered shapes as much as possible
                        for s in shapes:
                            for k in range(len(input_shape)):
                                if k != split_dim and is_symbolic_or_unknown(s[k]):
                                    s[k] = input_shape[k]
                                elif k == split_dim and is_symbolic_or_unknown(s[k]):
                                    s[k] = input_shape[k] // num_split
                        node.attr['split'] = [s[split_dim] for s in shapes]
                        from_shapes_ok = True
                except:
                    pass
                if not from_shapes_ok:
                    output_shape = list(input_shape[:])
                    idim = input_shape[split_dim]
                    if mode == 'Split':
                        assert (idim % num_split == 0)
                        if is_symbolic_or_known(idim):
                            node.attr['split'] = [idim // num_split] * num_split
                            output_shape[split_dim] = idim // num_split
                        else:
                            node.attr['split'] = [-1] * num_split
                            output_shape[split_dim] = -1
                        shapes = [output_shape] * num_split
                        try_materialize = True
                    else:
                        assert (np.sum(split_size) == idim or is_symbolic_or_unknown(idim))
                        node.attr['split'] = list(split_size)
                        shapes = [output_shape[:] for _ in range(len(split_size))]
                        for idx, s in enumerate(split_size):
                            shapes[idx][split_dim] = s

            types = [builtins.tensor(datatype, tuple(shape)) for shape in shapes]
        else:
            types = [
                builtins.tensor(datatype, tuple(shape)) for shape in node.attr['_output_shapes']
            ]
        rettype = builtins.tuple(types)
        if try_materialize:
            value = try_get_non_sym_val(self.gdict[node.inputs[value_idx]])
            if value is not None:
                node.attr["symbolic_value"] = rettype()
                node.attr["symbolic_value"].val = np.split(value, num_split, axis=split_dim)
        return rettype

    def visit_SplitV(self, node):
        # this is like split but has shapes
        # implemented in Split
        return self.visit_Split(node, mode='SplitV')

    def visit_MatrixBandPart(self, node):
        assert (len(node.inputs) == 3)
        return self.visit(node.inputs[0])

    def visit_Unpack(self, node):
        input_type = self.visit(node.inputs[0])
        input_shape = input_type.get_shape()
        axis = node.attr['axis']
        assert (dim > 0 for dim in input_shape[:axis])
        length = input_shape[axis]
        retshape = input_shape[:axis] + input_shape[axis + 1:]
        return builtins.tuple([builtins.tensor(input_type.get_primitive(), tuple(retshape))] *
                              length)

    def visit_StopGradient(self, node):
        # this is just identity
        node.op = 'Identity'
        return self._visit_unary(node)

    def visit_Mean(self, node):
        return self._visit_reduce(node)

    def visit_Squeeze(self, node):
        sourcetype = self.visit(node.inputs[0])
        if sourcetype is not None:
            assert builtins.is_tensor(sourcetype)  # only tensor is squeeze-able
            squeezed_shape = list(sourcetype.T[1])
            d = sorted(node.attr['squeeze_dims'])
            if len(d) > 0:
                d = d[::-1]
                for i in d:
                    squeezed_shape.pop(i)
            else:
                squeezed_shape = [s for s in squeezed_shape if s != 1]
            rettype = builtins.tensor(sourcetype.get_primitive(), tuple(squeezed_shape))
            if self.gdict[node.inputs[0]].attr['symbolic_value'] is not None:
                val = self.gdict[node.inputs[0]].attr['symbolic_value'].val
                retval = np.squeeze(val, axis=tuple(d))
                node.attr['symbolic_value'] = rettype()
                node.attr['symbolic_value'].val = retval
            return rettype
        datatype = self._get_type_from_attr(node)
        return datatype

    def _bitstring_to_reverse_indices(self, i):
        # returns indices in reverse order
        indices = []
        ctr = 0
        if isinstance(i, list):
            return i
        while (i > 0):
            if i % 2 == 1:
                indices.append(ctr)
            i = i // 2
            ctr += 1
        return indices

    def _isKthBitSet(self, n, k):
        if n & (1 << (k)):
            return True
        else:
            return False

    def visit_StridedSlice(self, node):
        # this is massively complicated
        # https://www.tensorflow.org/api_docs/python/tf/strided_slice
        for i in node.inputs:
            self.visit(i)
        input_type = self.visit(node.inputs[0])
        input_shape = input_type.get_shape()
        # unknown input shape. not common. should not happen really.
        if len(input_shape) == 0:
            return input_type

        input_value = self.gdict[node.inputs[0]].attr['symbolic_value']

        begin_value = self.gdict[node.inputs[1]].attr['symbolic_value']
        end_value = self.gdict[node.inputs[2]].attr['symbolic_value']
        stride_value = self.gdict[node.inputs[3]].attr['symbolic_value']

        # these masks here are really really complicated
        assert node.attr.get('new_axis_mask', 0) == 0

        if all([begin_value, end_value, stride_value]):
            input_rank = len(input_shape)
            num_spec = len(begin_value.val)
            assert input_rank >= num_spec

            dim = 0
            begin_mask, end_mask, shrink_axes = [], [], []
            begin_ids, end_ids, strides = [], [], []
            for spec_id in range(num_spec):
                if self._isKthBitSet(node.attr.get('ellipsis_mask', 0), spec_id):
                    num_ellipsis_dims = input_rank - num_spec + 1
                    for _ in range(num_ellipsis_dims):
                        begin_mask.append(dim)
                        end_mask.append(dim)
                        begin_ids.append(0)
                        end_ids.append(0)
                        strides.append(1)
                        dim += 1
                elif self._isKthBitSet(node.attr.get('shrink_axis_mask', 0), spec_id):
                    shrink_axes.append(dim)
                    begin_ids.append(begin_value.val[spec_id])
                    end_ids.append(end_value.val[spec_id])
                    strides.append(stride_value.val[spec_id])
                    dim += 1
                else:
                    if self._isKthBitSet(node.attr.get('begin_mask', 0), spec_id):
                        begin_mask.append(dim)

                    if self._isKthBitSet(node.attr.get('end_mask', 0), spec_id):
                        end_mask.append(dim)

                    begin_ids.append(begin_value.val[spec_id])
                    end_ids.append(end_value.val[spec_id])
                    strides.append(stride_value.val[spec_id])
                    dim += 1

            begin_value = builtins.tensor(begin_value.get_primitive(), (input_rank,))()
            begin_value.val = np.array(begin_ids)

            end_value   = builtins.tensor(end_value.get_primitive(), (input_rank,))()
            end_value.val = np.array(end_ids)

            stride_value = builtins.tensor(stride_value.get_primitive(), (input_rank,))()
            stride_value.val = np.array(strides)
        else:
            assert node.attr.get('ellipsis_mask', 0) == 0
            shrink_axes = self._bitstring_to_reverse_indices(node.attr.get('shrink_axis_mask', 0))
            begin_mask = self._bitstring_to_reverse_indices(node.attr.get('begin_mask', 0))
            end_mask = self._bitstring_to_reverse_indices(node.attr.get('end_mask', 0))

        # try to solve for value if possible
        output_value = None
        rettype = None
        if not None in [input_value, begin_value, end_value, stride_value]:
            begin = [int(i) for i in list(begin_value.val[:])]
            end = [int(i) for i in list(end_value.val[:])]
            for i in begin_mask:
                begin[i] = 0
            for i in end_mask:
                end[i] = None
            # Similar issue to https://github.com/tensorflow/tensorflow/issues/19260
            for i in shrink_axes:
                if begin[i] is None:
                    end[i] = 1
                elif begin[i] == -1:
                    end[i] = None
                else:
                    end[i] = begin[i] + 1
            slices = [slice(*i) for i in zip(begin, end, stride_value.val)]
            # insert missing slices
            for i in range(len(slices), len(input_shape)):
                slices.append(slice(None, None, None))

            slices = tuple(slices)
            res = input_value.val[slices]

            # remove shrink axes
            if len(shrink_axes) > 0:
                if len(shrink_axes) == len(res.shape):
                    if len(res) == 0:
                        logging.warning("%s:%s seems to be a 0 sized tensor", node.name, node.op)
                        return builtins.tensor(input_type.get_primitive(), [])
                    res = res.tolist()[0]
                else:
                    res = np.squeeze(res, axis=tuple(shrink_axes))
            # if we have a complete value, we can force it

            slicesv = [[begin[i], end[i], stride_value.val[i]] for i in range(len(begin))]
            for idx, s in enumerate(slicesv):
                if s[0] is None:
                    s[0] = 0
                    begin_mask.append(idx)
                if s[1] is None:
                    s[1] = 2147483647
                    end_mask.append(idx)
                if s[2] is None:
                    s[2] = 1
                s[0] = int(s[0])
                s[1] = int(s[1])
                s[2] = int(s[2])
            # insert missing slices
            for i in range(len(slicesv), len(input_shape)):
                slicesv.append([0, 2147483647, 1])
                if i not in begin_mask:
                    begin_mask.append(i)
                if i not in end_mask:
                    end_mask.append(i)
            node.attr['slice'] = slicesv
            node.attr['squeeze'] = list(int(i) for i in shrink_axes)
            node.attr['begin_masks'] = list(int(i) for i in begin_mask)
            node.attr['end_masks'] = list(int(i) for i in end_mask)
            if isscalar(res):
                rettype = input_type.get_primitive()
                output_value = rettype()
                output_value.val = res
            elif not isscalar(res):
                rettype = builtins.tensor(input_type.get_primitive(), res.shape)
                output_value = rettype()
                output_value.val = res

        # solve for type
        if rettype is None:
            # try to derive entirely from input_shape
            if (None in [begin_value, end_value, stride_value]):
                if len(input_shape) == len(shrink_axes):
                    # we are removing all axes. i.e. we are indexing a
                    # specific element
                    rettype = input_type.get_primitive()
                else:
                    new_shape = [
                        make_symbol(node.name + "_s_" + str(i))
                        for i in range(len(input_shape) - len(shrink_axes))
                    ]
                    rettype = builtins.tensor(input_type.get_primitive(), new_shape)
                # we have a non-constant shaped slice
                # store the sqeeze
                node.attr['squeeze'] = list(int(i) for i in shrink_axes)
                node.attr['begin_masks'] = list(int(i) for i in begin_mask)
                node.attr['end_masks'] = list(int(i) for i in end_mask)
            else:
                retshape = []
                begin = begin_value.val[:].tolist()
                end = end_value.val[:].tolist()
                begin = self._shape_as_ints(begin)
                end = self._shape_as_ints(end)
                for i in begin_mask:
                    begin[i] = None
                for i in end_mask:
                    end[i] = None
                for i in shrink_axes:
                    if begin[i] is None:
                        end[i] = 1
                    elif begin[i] == -1:
                        end[i] = None
                    else:
                        end[i] = begin[i] + 1
                if stride_value is not None:
                    stride_value = list(stride_value.val[:].astype(np.int32))

                for i in range(len(begin)):
                    if i in shrink_axes:
                        retshape.append(1)
                    elif is_symbolic_or_unknown(input_shape[i]):
                        if np.isscalar(begin[i]) and np.isscalar(
                                end[i]) and np.isscalar(stride_value):
                            retshape.append(len(list(range(begin[i], end[i], stride_value[i]))))
                        elif (is_symbolic_or_unknown(begin[i])
                              or is_symbolic_or_unknown(end[i])) and stride_value[i] == 1:
                            if end[i] is None:
                                retshape.append(input_shape[i] - begin[i])
                            else:
                                retshape.append(end[i] - begin[i])
                        else:
                            retshape.append(make_symbol(node.name + '_' + str(i)))
                    else:
                        if begin[i] is not None and begin[i] < 0:
                            try:
                                begin[i] += input_shape[i]
                            except:
                                pass
                        if end[i] is None:
                            end[i] = None # used to be input_shape[i]
                        elif end[i] < 0:
                            try:
                                end[i] += input_shape[i]
                            except:
                                pass
                        thisslice = slice(begin[i], end[i], stride_value[i])
                        thisslicelen = len(list(range(input_shape[i]))[thisslice])
                        retshape.append(thisslicelen)
                slices = [[begin[i], end[i], stride_value[i]] for i in range(len(begin))]
                has_symbolic_slices = False
                for idx, s in enumerate(slices):
                    if s[0] is None:
                        s[0] = 0
                        begin_mask.append(idx)
                    if s[1] is None:
                        s[1] = 2147483647
                        end_mask.append(idx)
                    if s[2] is None:
                        s[2] = 1
                    try:
                        s[0] = int(s[0])
                    except:
                        has_symbolic_slices = True
                        pass
                    try:
                        s[1] = int(s[1])
                    except:
                        has_symbolic_slices = True
                        pass
                    try:
                        s[2] = int(s[2])
                    except:
                        has_symbolic_slices = True
                        pass
                # insert missing slices
                for i in range(len(slices), len(input_shape)):
                    slices.append([0, 2147483647, 1])
                    retshape.append(input_shape[i])
                    if i not in begin_mask:
                        begin_mask.append(i)
                    if i not in end_mask:
                        end_mask.append(i)

                if not has_symbolic_slices:
                    node.attr['slice'] = slices
                node.attr['squeeze'] = list(int(i) for i in shrink_axes)
                node.attr['begin_masks'] = list(int(i) for i in begin_mask)
                node.attr['end_masks'] = list(int(i) for i in end_mask)
                # drop removed axes
                for a in shrink_axes:
                    assert (retshape[a] == 1 or is_symbolic_or_unknown(retshape[a]))
                retshape = [s for i, s in enumerate(retshape) if i not in shrink_axes]
                if len(retshape) == 0:
                    rettype = input_type.get_primitive()
                else:
                    rettype = builtins.tensor(input_type.get_primitive(), retshape)
        node.attr['symbolic_value'] = output_value
        return rettype

    def visit_Max(self, node):
        return self._visit_reduce(node)

    def visit_Min(self, node):
        return self._visit_reduce(node)

    def visit_Ceil(self, node):
        return self._visit_unary(node)

    def visit_Round(self, node):
        return self._visit_unary(node)

    def visit_Abs(self, node):
        return self._visit_unary(node)

    def visit_Floor(self, node):
        return self._visit_unary(node)

    def visit_Tile(self, node):
        if len(node.inputs) != 2:
            raise ValueError('Expected 2 inputs to {} node {}'.format(node.op, node.name))

        input_type = self.visit(node.inputs[0])
        tile_type = self.visit(node.inputs[1])

        if not builtins.is_tensor(input_type):
            raise ValueError('Expected tensor input to {} node {}'.format(node.op, node.name))

        if not builtins.is_tensor(tile_type) or len(tile_type.get_shape()) != 1:
            raise ValueError('Expected tensor tile input to {} node {}'.format(node.op, node.name))

        if tile_type.get_shape()[0] != len(input_type.get_shape()):
            raise ValueError(
                'Tile specification must be length of input rank to {} node {}'.format(
                    node.op, node.name))

        input_shape = input_type.get_shape()
        if len(input_shape) == 0:
            return input_type

        input_value = self._get_symbolic_value(node.inputs[0])
        if input_value is not None:
            input_value = input_value.val

        tile_value = self._get_symbolic_value(node.inputs[1])
        if tile_value is None:
            ret_shape = [make_symbol(node.name + "_" + str(i)) for i in range(len(input_shape))]
            return builtins.tensor(input_type.get_primitive(), ret_shape)
        tile_value = tile_value.val

        if len(tile_value) != len(input_shape):
            raise ValueError(
                'Tile specification value must be length of inpout rank to {} node {}'.format(
                    node.op, node.name))

        rettype = builtins.tensor(
            input_type.get_primitive(),
            [input_shape[i] * tile_value[i] for i in range(len(tile_value))])
        if input_value is not None and tile_value is not None and not any_symbolic_or_unknown(tile_value):
            node.attr['symbolic_value'] = rettype()
            node.attr['symbolic_value'].val = np.tile(input_value, tile_value)
        return rettype

    def visit_FloorDiv(self, node):
        return self._visit_broadcast(node)

    def visit_RealDiv(self, node):
        return self._visit_broadcast(node)

    def visit_OneHot(self, node):
        """
        Inputs:
            0: (str) Name of value indicating indicies to be "on".
            1: (str) Name of value indicating depth of the one-hot dimension, i.e. the
                number of values in the domain.
            2: (str) Name of value indicating the "on" value.
            3: (str) Name of value indicating the "off" value.
        Attributes:
            axis (Required int): axis to fill; -1 for a new inner-most axis.
            dtype (Optional any): presence indicates T should be used.
            T (Conditional builtin) primitive type of the output tensor.
        """
        if len(node.inputs) != 4:
            raise ValueError('Expected 4 inputs to {} node {}'.format(node.op, node.name))

        indices_type = self.visit(node.inputs[0])
        depth_type = self.visit(node.inputs[1])
        on_type = self.visit(node.inputs[2])
        off_type = self.visit(node.inputs[3])

        if not builtins.is_int(depth_type):
            raise ValueError('depth must be integral in {} node {}'.format(node.op, node.name))

        if not builtins.utils.is_primitive(on_type) or not builtins.utils.is_primitive(off_type):
            raise ValueError(
                'On and off types must be primitive in {} node {}'.format(node.op, node.name))

        if on_type != off_type:
            raise ValueError(
                'On and off types must be the same in {} node {}'.format(node.op, node.name))

        axis = node.attr.get('axis')
        if not isinstance(axis, six.integer_types) or axis < -1:
            raise ValueError('axis must be integer >= -1 in {} node {}'.format(node.op, node.name))

        if builtins.is_tensor(indices_type):
            indices_shape = list(indices_type.get_shape())
        else:
            indices_shape = [1]

        depth_value = self._get_symbolic_value(node.inputs[1]).val
        if depth_value is None:
            depth_value = make_symbol(node.name + '_depth')
        elif depth_value < 0:
            raise ValueError('depth must be non-negative in {} node {}'.format(node.op, node.name))

        if 'dtype' in node.attr:
            ret_primitive = node.attr.get('T')
            if ret_primitive is None or not builtins.is_primitive(ret_primitive):
                raise ValueError(
                    'Output tensor data type must be primitive in {} node {}'.format(
                        node.op, node.name))
        else:
            ret_primitive = on_type

        if len(indices_shape) == 0:
            return builtins.tensor(ret_primitive, tuple())
        retshape = indices_shape
        if axis == -1:
            retshape.append(depth_value)
        else:
            retshape.insert(axis, depth_value)
        return builtins.tensor(ret_primitive, retshape)

    def visit_SquaredDifference(self, node):
        return self._visit_broadcast(node)

    def visit_Sub(self, node):
        return self._visit_broadcast(node)

    def visit_Sum(self, node):
        return self._visit_reduce(node)

    def visit_Tanh(self, node):
        return self._visit_unary(node)

    def find_tensor_array_source_node(self, node):
        if 'tensorarray_source' in node.attr:
            loc = node.attr['tensorarray_source']
            if loc in self.whole_ssa.global_resource:
                return self.whole_ssa.global_resource[loc]
        elif '_class' in node.attr:
            loc = node.attr['_class'][0][5:]
            if loc in self.whole_ssa.global_resource:
                return self.whole_ssa.global_resource[loc]

        return None

    def propagate_tensor_array(self, node):
        if node.op == 'make_tuple':
            tensorarray_source = [
                self.gdict[i].attr.get('tensorarray_source', None) for i in node.inputs
            ]
            node.attr['tensorarray_source'] = tensorarray_source
        elif node.op == 'get_tuple':
            if 'tensorarray_source' in self.gdict[node.inputs[0]].attr:
                tensorarray_source = self.gdict[node.inputs[0]].attr['tensorarray_source'][
                    node.attr['index']]
                node.attr['tensorarray_source'] = tensorarray_source
        else:
            self.visit(node.inputs[-1])
            if 'tensorarray_source' in self.gdict[node.inputs[-1]].attr:
                node.attr['tensorarray_source'] = self.gdict[
                    node.inputs[-1]].attr['tensorarray_source']

    def visit_TensorArrayV3(self, node):
        # input is size
        assert (len(node.inputs) <= 1)
        self.visit(node.inputs[0])
        # the input is an int32 which is the size of the tensor
        sizeval = self.gdict[node.inputs[0]].attr['symbolic_value']

        if sizeval is not None and not node.attr['dynamic_size']:
            assert isscalar(sizeval.val)
            node.attr['size'] = sizeval.val

        if 'infer_shape' in node.attr:
            # We only support fix size of TensorArray.
            assert (node.attr['infer_shape'])

        if isinstance(node.attr['element_shape'], six.string_types):
            val = self.resolve_to_non_sym_val_or_die(node.attr["element_shape"])
            node.attr["element_shape"] = list(val)

        if isinstance(node.attr.get('element_shape', []), list):
            shape = []
            if 'element_shape' in node.attr:
                shape = node.attr['element_shape']
            node.attr['element_shape'] = builtins.tensor(node.attr['dtype'], shape)
        self.whole_ssa.global_resource[node.name] = node
        node.attr['tensorarray_source'] = node.name

        return builtins.list(node.attr['element_shape']) if node.attr['element_shape'] else None

    def visit_TensorArrayGatherV3(self, node):
        # input is resource, indices, flow
        assert (len(node.inputs) == 2)
        indices_type = self.visit(node.inputs[0])

        self.propagate_tensor_array(node)
        tanode = self.find_tensor_array_source_node(node)

        if isinstance(node.attr['element_shape'], six.string_types):
            val = self.resolve_to_non_sym_val_or_die(node.attr["element_shape"])
            node.attr["element_shape"] = list(val)

        if indices_type is None:
            return builtins.tensor(tanode.attr['dtype'], [-1] + node.attr['element_shape'])
        else:
            indiceslen = indices_type.get_shape()[0]
            return builtins.tensor(tanode.attr['dtype'], [indiceslen] + node.attr['element_shape'])

    def visit_TensorArrayReadV3(self, node):
        # input is resource, idx, flow
        assert (len(node.inputs) == 2)

        self.propagate_tensor_array(node)
        tanode = self.find_tensor_array_source_node(node)

        ta_type = self.visit(node.inputs[1])

        if tanode is not None:
            ta_type = tanode.datatype
        return ta_type.T[0] if ta_type else None

    def visit_TensorArrayScatterV3(self, node):
        # input is resource, indices, values , flow
        self.propagate_tensor_array(node)
        tanode = self.find_tensor_array_source_node(node)

        tensor_put_type = self.visit(node.inputs[1])
        assert (builtins.is_tensor(tensor_put_type))
        tensor_put_type = builtins.tensor(
            tensor_put_type.get_primitive(),
            tensor_put_type.get_shape()[1:])

        # Overide the shape in the node attributes
        if len(tensor_put_type.get_shape()) > 0 and tanode is not None:
            el_shape = tanode.attr.get('element_shape')
            es = None if el_shape is None else el_shape.get_shape()
            if (es is None or len(es) == 0 \
                or (-1 in es and -1 not in tensor_put_type.get_shape())):
                tanode.attr['element_shape'] = tensor_put_type

        # output is flow
        assert (len(node.inputs) == 3)
        return self.visit(node.inputs[2])

    def visit_TensorArraySizeV3(self, node):

        self.propagate_tensor_array(node)
        tanode = self.find_tensor_array_source_node(node)
        for inputnodes in node.inputs:
            self.visit(inputnodes)

        if tanode is not None and 'size' in tanode.attr and not tanode.attr.get('dynamic_size',
                                                                                True):
            node.attr['symbolic_value'] = builtins.int32()
            node.attr['symbolic_value'].val = tanode.attr['size']

        # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/ops/data_flow_ops.cc
        return builtins.int32

    def visit_TensorArrayWriteV3(self, node):
        # input is resource, index, value, flow
        # output is flow
        # try to infer tensor array element type if possible
        self.propagate_tensor_array(node)
        tanode = self.find_tensor_array_source_node(node)

        tensor_put_type = self.visit(node.inputs[1])
        # Overide the shape in the node attributes

        if hasattr(tensor_put_type, 'get_shape') and \
            len(tensor_put_type.get_shape()) > 0 and tanode is not None:
            el_shape = tanode.attr.get('element_shape')
            es = None if el_shape is None else el_shape.get_shape()
            if (es is None or len(es) == 0 \
                or (-1 in es and -1 not in tensor_put_type.get_shape())):
                tanode.attr['element_shape'] = tensor_put_type

        assert (len(node.inputs) == 3)
        return self.visit(node.inputs[2])

    def visit_TopKV2(self, node):
        """
        Inputs:
            0 (str): The name of a tensor
            1 (str): The name of an int32
        """
        if len(node.inputs) != 2:
            raise ValueError('Expected 2 inputs to {} node {}'.format(node.op, node.name))

        input_type = self.visit(node.inputs[0])
        k_type = self.visit(node.inputs[1])
        k_val = self._get_symbolic_value(node.inputs[1])

        if not builtins.is_tensor(input_type):
            raise ValueError('Input must be a tensor in {} node {}'.format(node.op, node.name))

        if not builtins.is_int(k_type):
            raise ValueError('K must be an int in {} node {}'.format(node.op, node.name))

        if k_val is not None:
            k = k_val.val
            if not is_symbolic(k) and k > input_type.get_shape()[-1]:
                raise ValueError(
                    'K greater than size of last dimension in {} node {}'.format(
                        node.op, node.name))
        else:
            k = make_symbol(node.name + '_k')

        ret_shape = list(input_type.get_shape())
        ret_shape[-1] = k
        return builtins.tuple((
            builtins.tensor(input_type.get_primitive(),
                            ret_shape), builtins.tensor(builtins.int32, ret_shape)))

    def visit_Transpose(self, node):
        """
        Inputs:
            0 (str): Name of an input tensor.
            1 (str): Name of a 1-D tensor indicating how to permute the input.
        """
        if len(node.inputs) != 2:
            raise ValueError('Expected 2 inputs to {} node {}'.format(node.op, node.name))

        inputtype = self.visit(node.inputs[0])
        permtype = self.visit(node.inputs[1])

        if not builtins.is_tensor(inputtype):
            raise ValueError('Input must be a tensor in {} node {}'.format(node.op, node.name))

        shape = inputtype.get_shape()
        primitive = inputtype.get_primitive()

        if not builtins.is_tensor(permtype) or len(shape) != permtype.get_shape()[0]:
            raise ValueError(
                'Permutation must be a 1-D tensor as long as input rank in {} node {}'.format(
                    node.op, node.name))

        transpose_axes = self._get_symbolic_value(node.inputs[1])

        if transpose_axes is None or transpose_axes.val is None:
            # We can't determine the output shape right now: figure it out at runtime
            shape = tuple(make_symbol(node.name + str(i)) for i in range(len(shape)))
            rettype = builtins.tensor(primitive, shape)
            return rettype

        if len(transpose_axes.val) != len(shape):
            raise ValueError(
                'Permutation symbolic value must be a 1-D tensor as long as input rank in {} node {}'
                .format(node.op, node.name))

        # Compute the output shape
        new_shape = []
        for ax in transpose_axes.val:
            new_shape.append(shape[ax])
        rettype = builtins.tensor(primitive, new_shape)

        # Compute a symbolic value for this node if possible
        input0 = try_get_non_sym_val(self.gdict[node.inputs[0]])
        input1 = try_get_non_sym_val(self.gdict[node.inputs[1]])
        if input0 is not None and input1 is not None:
            self._set_symbolic_value(node, rettype, np.transpose(input0, axes=input1))

        return rettype

    def visit_VariableV2(self, node):
        return None

    def visit_while(self, node):
        assert ("cond_function" in node.attr)
        assert ("body_function" in node.attr)
        assert (len(node.inputs) == 1)

        mytype = self.visit(node.inputs[0])

        functions_called = [node.attr[i] for i in ["cond_function", "body_function"]]
        for f in functions_called:
            if self.whole_ssa is not None and f in self.whole_ssa.functions:
                # look for the function entry point
                entrypoint = [
                    n for n in self.whole_ssa.functions[f].graph.values()
                    if n.op == 'function_entry'
                ]
                entrypoint[0].datatype = mytype
                if 'tensorarray_source' in self.gdict[node.inputs[0]].attr:
                    entrypoint[0].attr['tensorarray_source'] = self.gdict[
                        node.inputs[0]].attr['tensorarray_source']
        if 'tensorarray_source' in self.gdict[node.inputs[0]].attr:
            node.attr['tensorarray_source'] = self.gdict[node.inputs[0]].attr['tensorarray_source']

        return mytype

    def visit_get_global(self, node):
        assert ("variable" in node.attr)
        assert (node.attr['variable'] in self.whole_ssa.variables)
        return self.whole_ssa.variables[node.attr['variable']].__class__

    def visit_set_global(self, node):
        assert ("variable" in node.attr)
        assert (node.attr['variable'] in self.whole_ssa.variables)
        input_type = self.visit(node.inputs[0])
        variable_type = self.whole_ssa.variables[node.attr['variable']].__class__
        if input_type is not None:
            if not (input_type is variable_type
                    or builtins.is_tensor_and_is_compatible_general_shape(input_type,
                                                                          variable_type)[0]):
                logging.warning(
                    "Possible incompatible type in set_global: %s. expected %s",
                    builtins.get_type_info(input_type), builtins.get_type_info(variable_type))
        return input_type

    def visit_Size(self, node):
        self.visit(node.inputs[0])
        parenttype = self.gdict[node.inputs[0]].datatype
        rettype = node.attr["out_type"]
        if parenttype is not None:
            input_shape = parenttype.get_shape()
            node.attr['symbolic_value'] = rettype()
            node.attr['symbolic_value'].val = np.prod(input_shape)
        return rettype

    def visit_Sign(self, node):
        input_type = self.visit(node.inputs[0])
        return input_type

    def visit_Cumsum(self, node):
        assert (len(node.inputs) == 2)
        return self.visit(node.inputs[0])

    def visit_ClipByValue(self, node):
        assert len(node.inputs) == 3

        type_min = self.visit(node.inputs[1])
        type_max = self.visit(node.inputs[2])
        if not (builtins.is_tensor(type_max) or builtins.is_tensor(type_min)):
            node.attr["min_value"] = self.gdict[node.inputs[1]].attr['value'].val
            node.attr["max_value"] = self.gdict[node.inputs[2]].attr['value'].val

        return self.visit(node.inputs[0])

    def visit_SpaceToDepth(self, node):
        return self._get_type_from_attr(node)

    def visit_DepthToSpace(self, node):
        return self._get_type_from_attr(node)

    def visit_SpaceToBatchND(self, node):
        return self._get_type_from_attr(node)

    def visit_BatchToSpaceND(self, node):
        return self._get_type_from_attr(node)

    def visit_LRN(self, node):
        return self._visit_unary(node)

    def visit_Reciprocal(self, node):
        return self._visit_unary(node)


def type_is_unknown(t):
    if builtins.is_tuple(t):
        return any(type_is_unknown(a) for a in t.T)
    elif builtins.is_tensor(t):
        return type_is_unknown(t.get_primitive()) or \
               t.get_shape() is None or \
               len(t.get_shape()) == 0 or \
               any_symbolic_or_unknown(t.get_shape())
    elif builtins.is_list(t):
        return type_is_unknown(t.T[0])
    elif t is builtins.unknown:
        return True
    else:
        return t is None


def type_inference_pass_impl(nnssa):
    """
    Takes an NetworkEnsemble object and performs recursive type inference
    on all the nodes in the graph
    """
    function_names = list(nnssa.functions.keys())
    function_names = sorted(function_names)
    # stick the main functions at the start
    if "main" in function_names:
        function_names = ["main"] + [i for i in function_names if i != "main"]

    import copy
    # try to infer all the set_global types first
    changed_variables = []
    for k in function_names:
        graph = copy.copy(nnssa.functions[k].graph)
        for v in graph.values():
            if v.op == 'set_global':
                rettype = TypeInferenceVisitor(graph, nnssa).visit(v)
                variable = v.attr['variable']
                validate_shape =  v.attr.get('validate_shape', True)
                if (variable in changed_variables) and validate_shape:
                    if builtins.get_type_info(
                            nnssa.variables[variable]) == builtins.get_type_info(rettype):
                        continue
                    else:
                        raise TypeError(
                            "Varable %s changes type several times from %s to %s" % (
                                variable, builtins.get_type_info(
                                    nnssa.variables[variable]), builtins.get_type_info(rettype)))
                if rettype != type(nnssa.variables[variable]):
                    nnssa.variables[variable] = rettype()
                    if variable not in changed_variables:
                        changed_variables.append(variable)
                    logging.debug(
                        "Changing variable %s to type %s", variable,
                        builtins.get_type_info(rettype))
        nnssa.functions[k].find_inputs_and_outputs()

    # reinfer unknown shapes and types
    for k in function_names:
        graph = copy.copy(nnssa.functions[k].graph)
        for v in graph.values():
            if type_is_unknown(v.datatype):
                v.datatype = None

    # run it for real
    for k in function_names:
        TypeInferenceVisitor(nnssa.functions[k].graph, nnssa).visit_all()


def recursive_replace_symbols_in_type_with_unknown(dtype):
    if builtins.is_list(dtype):
        return builtins.list(recursive_replace_symbols_in_type_with_unknown(dtype.T[0]))
    elif builtins.is_tuple(dtype):
        return builtins.tuple(
            tuple(recursive_replace_symbols_in_type_with_unknown(t) for t in dtype.T))
    elif builtins.is_tensor(dtype):
        return builtins.tensor(
            dtype.get_primitive(),
            tuple(-1 if issubclass(type(t), sm.Basic) else int(t) for t in dtype.get_shape()))
    else:
        return dtype


def recursive_replace_symbols_in_values(val):
    # try some things in sympy.core.numbers
    if issubclass(type(val), sm.Basic):
        return int(val)
    elif isinstance(val, list):
        return [recursive_replace_symbols_in_values(i) for i in val]
    elif isinstance(val, tuple):
        return tuple([recursive_replace_symbols_in_values(i) for i in val])
    elif isinstance(val, np.ndarray):
        if np.issctype(val.dtype):
            return val
        else:
            return np.array([recursive_replace_symbols_in_values(i)
                             for i in val.flatten()]).reshape(val.shape)
    else:
        return val


def graph_replace_symbolic_values(gdict):
    for k in gdict:
        v = gdict[k]
        if v.value is None and v.attr['symbolic_value'] is not None and not any_symbolic_or_unknown(
                v.attr['symbolic_value'].val):
            v.value = v.attr['symbolic_value']
            v.value.val = recursive_replace_symbols_in_values(v.value.val)
        v.attr['symbolic_datatype'] = v.datatype
        v.datatype = recursive_replace_symbols_in_type_with_unknown(v.datatype)


def graph_make_symbolic_values(gdict):
    for k in gdict:
        gdict[k].attr['symbolic_value'] = gdict[k].value


def type_inference_pass(nnssa):
    # repeat for as many times as there are functions
    # this is the maximum number of times required for convergence
    for i in nnssa.functions:
        graph_make_symbolic_values(nnssa.functions[i].graph)
    for i in range(len(nnssa.functions)):
        type_inference_pass_impl(nnssa)
    for i in nnssa.functions:
        graph_replace_symbolic_values(nnssa.functions[i].graph)
    for i in nnssa.variables:
        nnssa.variables[i] = recursive_replace_symbols_in_type_with_unknown(nnssa.variables[i])
