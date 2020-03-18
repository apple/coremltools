from coremltools.converters.nnv2.builtin_types.symbolic import is_symbolic
from ._op_reqs import *

def _broadcast_shapes(shape_x, shape_y):
    """
    Check and broadcast given input shapes.
    :param shape_x: tuple of int or symbols
        Shape of the first tensor (possibly symbolic).
    :param shape_y: tuple of int or symbols
        Shape of the second tensor (possibly symbolic).
    :return: tuple of int or symbols
        Result from broadcast.
    """
    if len(shape_x) < len(shape_y):
        shape_x = ([1] * (len(shape_y) - len(shape_x))) + shape_x
    if len(shape_y) < len(shape_x):
        shape_y = ([1] * (len(shape_x) - len(shape_y))) + shape_y

    ret_shapes = list()
    for i in range(len(shape_x)):
        x_unknown = is_symbolic(shape_x[i])
        y_unknown = is_symbolic(shape_y[i])
        if shape_x[i] == 1:
            ret_shapes.append(shape_y[i])
        elif shape_y[i] == 1:
            ret_shapes.append(shape_x[i])
        elif not y_unknown and shape_y[i] > 1:
            if not x_unknown and shape_x[i] != shape_y[i]:
                raise ValueError(
                    'Incompatible dim {} in shapes {} vs. {}'.format(
                        i, shape_x, shape_y))
            ret_shapes.append(shape_y[i])
        elif not x_unknown and shape_x[i] > 1:
            if not y_unknown and shape_x[i] != shape_y[i]:
                raise ValueError(
                    'Incompatible dim {} in shapes {} vs. {}'.format(
                        i, shape_x, shape_y))
            ret_shapes.append(shape_x[i])
        elif x_unknown or y_unknown:
            ret_shapes.append(sm.functions.Max(shape_x[i], shape_y[i]))
        else:
            assert (shape_x[i] == shape_y[i])
            ret_shapes.append(shape_x[i])

    return tuple(ret_shapes)


def _promoted_primitive_type(type1, type2):
    """
    Given a pair of tensor or primitive types, find the smallest type that can store an instance
    of their primitive type.
    """
    ptype1 = type1.get_primitive() if builtins.is_tensor(type1) else type1
    ptype2 = type2.get_primitive() if builtins.is_tensor(type2) else type2
    return builtins.promote_types(ptype1, ptype2)
