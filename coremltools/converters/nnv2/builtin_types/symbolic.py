import sympy as sm
import numpy as np
import six


def is_symbolic(val):
    return issubclass(type(val), sm.Basic)  # pylint: disable=consider-using-ternary

def is_variadic(val):
    return issubclass(type(val), sm.Symbol) and val.name[0] == '*'  # pylint: disable=consider-using-ternary

def num_symbolic(val):
    """
    Return the number of symbols in val
    """
    if is_symbolic(val):
        return 1
    elif isinstance(val, np.ndarray) and np.issctype(val.dtype):
        return 0
    elif hasattr(val, '__iter__'):
        return sum(any_symbolic(i) for i in val)
    return 0

def any_symbolic(val):
    if is_symbolic(val):
        return True
    elif isinstance(val, np.ndarray) and np.issctype(val.dtype):
        return False
    elif isinstance(val, six.string_types): # string is iterable
        return False
    elif hasattr(val, '__iter__'):
        return any(any_symbolic(i) for i in val)
    return False

def any_variadic(val):
    if is_variadic(val):
        return True
    elif isinstance(val, np.ndarray) and np.issctype(val.dtype):
        return False
    elif isinstance(val, six.string_types): # string is iterable
        return False
    elif hasattr(val, '__iter__'):
        return any(any_variadic(i) for i in val)
    return False

def isscalar(val):
    return np.isscalar(val) or issubclass(type(val), sm.Basic)
