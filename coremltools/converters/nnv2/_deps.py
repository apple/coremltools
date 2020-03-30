# Copyright (c) 2017, Apple Inc. All rights reserved.
#
"""
List of all external dependencies for this package. Imported as
optional includes
"""
from packaging import version

HAS_GRAPHVIZ = True
try:
    import graphviz
except:
    HAS_GRAPHVIZ = False

HAS_ONNX = True
try:
    import onnx
except:
    HAS_ONNX = False

HAS_TF1 = True
try:
    import tensorflow

    assert version.parse(tensorflow.__version__).release[0] < 2
except:
    HAS_TF1 = False

HAS_TF2 = True
try:
    import tensorflow

    assert version.parse(tensorflow.__version__).release[0] >= 2
except:
    HAS_TF2 = False

MSG_TF1_NOT_FOUND = 'TensorFlow 1.x not found.'
MSG_TF2_NOT_FOUND = 'TensorFlow 2.x not found.'

HAS_PYTORCH = True
try:
    import torch
except:
    HAS_PYTORCH = False
