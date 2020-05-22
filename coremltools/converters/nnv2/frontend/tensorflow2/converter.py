from coremltools.converters.nnv2.frontend.tensorflow.converter import *

from .ssa_passes.tf_passes import tensorflow_passes as tensorflow2_passes
from .ops import *  # register all the ops

from coremltools.converters.nnv2.frontend.tensorflow import converter

# TF2 uses different set of graph passes then TF1
converter.tensorflow_passes = tensorflow2_passes
