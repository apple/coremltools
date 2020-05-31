import numpy as np

from coremltools.converters.mil.mil import types
from coremltools.converters.mil.mil import Operation, precondition, VALUE
from coremltools.converters.mil.mil.input_type import *
from coremltools.converters.mil.mil.ops.registry import SSAOpRegistry

register_op = SSAOpRegistry.register_op
