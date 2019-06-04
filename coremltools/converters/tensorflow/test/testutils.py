# -*- coding: utf-8 -*-
from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _

import numpy as np


def generate_data(shape, mode='random'):
    """ Generate some random data according to a shape.
    """
    if shape is None or len(shape) == 0:
        return 0.5
    if mode == 'zeros':
        X = np.zeros(shape)
    elif mode == 'ones':
        X = np.ones(shape)
    elif mode == 'linear':
        X = np.array(range(np.product(shape))).reshape(shape) * 1.0
    elif mode == 'random':
        X = np.random.rand(*shape)
    elif mode == 'random_zero_mean':
        X = np.random.rand(*shape) - 0.5
    return X


def tf_transpose(x, channel_last=False):
    if not hasattr(x, "shape"):
        return np.array([x], dtype='float')
    elif len(x.shape) == 0:
        return np.array(x, dtype='float')
    elif len(x.shape) == 1:
        return x
    elif len(x.shape) == 2:
        return x
    elif len(x.shape) == 3:
        return x
    elif len(x.shape) == 4:
        if channel_last:
            return np.transpose(x, (0, 3, 1, 2))
        else:
            return x
    else:
        raise ValueError('tf_transpose not supporting x.shape = %s' % (str(x.shape)))
