# Copyright (c) 2017, Apple Inc. All rights reserved.
#
"""
List of all external dependencies for this package. Imported as
optional includes
"""

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

HAS_PYESPRESSO = True
try:
    import pyespresso
except:
    HAS_PYESPRESSO = False

HAS_TF = True
try:
    import tensorflow
except:
    HAS_TF = False

HAS_PYTORCH = True
try:
    import torch
except:
    HAS_PYTORCH = False
