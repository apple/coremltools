### Module for proto generated Python code.

import os as _os

if int(_os.environ.get("IMPORT_COREMLTOOLS_PROTO", "1")):
        from . import FeatureTypes_pb2, MIL_pb2, Model_pb2, NeuralNetwork_pb2
