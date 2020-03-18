import numpy as np

from coremltools.converters.nnv2.builtin_types import builtins
from coremltools.converters.nnv2.nnv2_program.program.program import Operation, precondition, VALUE
from coremltools.converters.nnv2.nnv2_program.program.input_type import *
from coremltools.converters.nnv2.nnv2_program.ops.registry import SSAOpRegistry

register_op = SSAOpRegistry.register_op
