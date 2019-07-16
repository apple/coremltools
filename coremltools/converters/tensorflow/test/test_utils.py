# -*- coding: utf-8 -*-
from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _

import numpy as np


def generate_data(shape, mode='random_zero_mean'):
    """
    Generate some random data according to a shape.
    """
    if shape is None or len(shape) == 0:
        return 0.5
    if mode == 'zeros':
        x = np.zeros(shape)
    elif mode == 'ones':
        x = np.ones(shape)
    elif mode == 'linear':
        x = np.array(range(np.product(shape))).reshape(shape) * 1.0
    elif mode == 'random':
        x = np.random.rand(*shape)
    elif mode == 'random_large':
        x = np.random.rand(*shape) * 100.0
    elif mode == 'random_int':
        x = np.random.randint(-50, 50, shape) * 1.0
    elif mode == 'random_zero_mean':
        x = np.random.rand(*shape) - 0.5
    elif mode == 'random_zeros_ones':
        x = np.random.randint(0, 2, shape) * 1.0
    elif mode == 'random_zero_mean_with_zeros':
        x = [np.random.choice([np.random.rand(), 0.0]) for _ in range(np.product(shape))]
        x = np.array(x).reshape(shape)
    else:
        raise ValueError("invalid data mode: '{}'.".format(mode))
    return x


def tf_transpose(x, channel_last=False):
    if not hasattr(x, 'shape'):
        return np.array([x], dtype=np.float)
    elif len(x.shape) == 0:
        return np.array(x, dtype=np.float)
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
        raise ValueError('tf_transpose does not support shape = {}'.format(x.shape))
