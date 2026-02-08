# Copyright (c) 2025, Apple Inc. All rights reserved.
#
# Use of this source code is governed by a BSD-3-clause license that can be
# found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

"""
Memory Estimator for Core ML Models.

Provides memory footprint analysis for Core ML models, including
parameter memory, activation memory, and total runtime memory estimation.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from coremltools.models import MLModel


DTYPE_BYTES = {
    "float32": 4,
    "float16": 2,
    "int32": 4,
    "int16": 2,
    "int8": 1,
    "uint8": 1,
    "bool": 1,
    "fp32": 4,
    "fp16": 2,
    "int64": 8,
}


@dataclass
class MemoryBreakdown:
    """Memory breakdown for a Core ML model."""
    
    total_bytes: int
    parameter_bytes: int
    activation_bytes: int
    overhead_bytes: int
    per_layer_bytes: Dict[str, int] = field(default_factory=dict)
    
    @property
    def total_mb(self) -> float:
        """Total memory in megabytes."""
        return self.total_bytes / (1024 * 1024)
    
    @property
    def parameter_mb(self) -> float:
        """Parameter memory in megabytes."""
        return self.parameter_bytes / (1024 * 1024)
    
    @property
    def activation_mb(self) -> float:
        """Activation memory in megabytes."""
        return self.activation_bytes / (1024 * 1024)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "total_bytes": self.total_bytes,
            "total_mb": self.total_mb,
            "parameter_bytes": self.parameter_bytes,
            "parameter_mb": self.parameter_mb,
            "activation_bytes": self.activation_bytes,
            "activation_mb": self.activation_mb,
            "overhead_bytes": self.overhead_bytes,
            "per_layer_bytes": self.per_layer_bytes,
        }


class MemoryEstimator:
    """
    Estimator for Core ML model memory footprint.
    
    Calculates the memory requirements for a Core ML model including:
    - Parameter memory (weights, biases)
    - Activation memory (intermediate tensors)
    - Runtime overhead
    
    Parameters
    ----------
    model : MLModel or str
        A loaded MLModel instance or path to a .mlmodel/.mlpackage file.
    
    Examples
    --------
    >>> import coremltools as ct
    >>> from coremltools.models.complexity import MemoryEstimator
    >>> 
    >>> model = ct.models.MLModel("my_model.mlpackage")
    >>> estimator = MemoryEstimator(model)
    >>> breakdown = estimator.estimate()
    >>> print(f"Total memory: {breakdown.total_mb:.2f} MB")
    >>> print(f"Parameters: {breakdown.parameter_mb:.2f} MB")
    """
    
    def __init__(self, model):
        if isinstance(model, str):
            self._model = MLModel(model)
        else:
            self._model = model
        
        self._spec = self._model.get_spec()
        self._breakdown: Optional[MemoryBreakdown] = None
    
    def estimate(self) -> MemoryBreakdown:
        """
        Estimate memory requirements for the model.
        
        Returns
        -------
        MemoryBreakdown
            Detailed memory breakdown.
        """
        parameter_bytes = 0
        activation_bytes = 0
        per_layer_bytes: Dict[str, int] = {}
        
        if self._spec.WhichOneof("Type") == "mlProgram":
            parameter_bytes, activation_bytes, per_layer_bytes = self._analyze_ml_program()
        elif self._spec.WhichOneof("Type") in ("neuralNetwork", "neuralNetworkClassifier", 
                                                 "neuralNetworkRegressor"):
            parameter_bytes, activation_bytes, per_layer_bytes = self._analyze_neural_network()
        
        overhead_bytes = int((parameter_bytes + activation_bytes) * 0.1)
        total_bytes = parameter_bytes + activation_bytes + overhead_bytes
        
        self._breakdown = MemoryBreakdown(
            total_bytes=total_bytes,
            parameter_bytes=parameter_bytes,
            activation_bytes=activation_bytes,
            overhead_bytes=overhead_bytes,
            per_layer_bytes=per_layer_bytes,
        )
        
        return self._breakdown
    
    def get_parameter_count(self) -> int:
        """
        Get total number of parameters in the model.
        
        Returns
        -------
        int
            Total parameter count.
        """
        total_params = 0
        
        if self._spec.WhichOneof("Type") == "mlProgram":
            total_params = self._count_ml_program_params()
        elif self._spec.WhichOneof("Type") in ("neuralNetwork", "neuralNetworkClassifier", 
                                                 "neuralNetworkRegressor"):
            total_params = self._count_neural_network_params()
        
        return total_params
    
    def _analyze_ml_program(self) -> Tuple[int, int, Dict[str, int]]:
        """Analyze memory for ML Program format models."""
        parameter_bytes = 0
        activation_bytes = 0
        per_layer: Dict[str, int] = {}
        
        program = self._spec.mlProgram
        
        for func in program.functions.values():
            for block_name, block in func.block_specializations.items():
                for op in block.operations:
                    op_name = op.outputs[0].name if op.outputs else "unknown"
                    op_param_bytes = self._estimate_op_parameter_bytes(op)
                    op_activation_bytes = self._estimate_op_activation_bytes(op)
                    
                    parameter_bytes += op_param_bytes
                    activation_bytes += op_activation_bytes
                    per_layer[op_name] = op_param_bytes + op_activation_bytes
        
        return parameter_bytes, activation_bytes, per_layer
    
    def _analyze_neural_network(self) -> Tuple[int, int, Dict[str, int]]:
        """Analyze memory for Neural Network format models."""
        parameter_bytes = 0
        activation_bytes = 0
        per_layer: Dict[str, int] = {}
        
        nn_spec = None
        if self._spec.HasField("neuralNetwork"):
            nn_spec = self._spec.neuralNetwork
        elif self._spec.HasField("neuralNetworkClassifier"):
            nn_spec = self._spec.neuralNetworkClassifier
        elif self._spec.HasField("neuralNetworkRegressor"):
            nn_spec = self._spec.neuralNetworkRegressor
        
        if nn_spec is None:
            return parameter_bytes, activation_bytes, per_layer
        
        for layer in nn_spec.layers:
            layer_param_bytes = self._estimate_nn_layer_parameter_bytes(layer)
            layer_activation_bytes = self._estimate_nn_layer_activation_bytes(layer)
            
            parameter_bytes += layer_param_bytes
            activation_bytes += layer_activation_bytes
            per_layer[layer.name] = layer_param_bytes + layer_activation_bytes
        
        return parameter_bytes, activation_bytes, per_layer
    
    def _estimate_op_parameter_bytes(self, op) -> int:
        """Estimate parameter bytes for a single operation."""
        param_bytes = 0
        
        if op.type == "const":
            for out in op.outputs:
                if out.HasField("type"):
                    shape = self._type_to_shape(out.type)
                    dtype = self._type_to_dtype(out.type)
                    if shape:
                        elements = int(np.prod([d for d in shape if d > 0]))
                        bytes_per_element = DTYPE_BYTES.get(dtype, 4)
                        param_bytes += elements * bytes_per_element
        
        return param_bytes
    
    def _estimate_op_activation_bytes(self, op) -> int:
        """Estimate activation bytes for a single operation."""
        activation_bytes = 0
        
        if op.type == "const":
            return 0
        
        for out in op.outputs:
            if out.HasField("type"):
                shape = self._type_to_shape(out.type)
                dtype = self._type_to_dtype(out.type)
                if shape and any(d > 0 for d in shape):
                    elements = int(np.prod([d for d in shape if d > 0]))
                    bytes_per_element = DTYPE_BYTES.get(dtype, 4)
                    activation_bytes += elements * bytes_per_element
        
        return activation_bytes
    
    def _estimate_nn_layer_parameter_bytes(self, layer) -> int:
        """Estimate parameter bytes for a Neural Network layer."""
        layer_type = layer.WhichOneof("layer")
        
        if layer_type == "convolution":
            conv = layer.convolution
            kernel_size = 1
            if conv.kernelSize:
                kernel_size = int(np.prod(conv.kernelSize))
            params = conv.outputChannels * conv.kernelChannels * kernel_size
            return params * 4
        
        elif layer_type == "innerProduct":
            ip = layer.innerProduct
            params = ip.inputChannels * ip.outputChannels
            if ip.hasBias:
                params += ip.outputChannels
            return params * 4
        
        elif layer_type == "batchnorm":
            bn = layer.batchnorm
            params = bn.channels * 4
            return params * 4
        
        return 0
    
    def _estimate_nn_layer_activation_bytes(self, layer) -> int:
        """Estimate activation bytes for a Neural Network layer."""
        return 4096
    
    def _count_ml_program_params(self) -> int:
        """Count parameters in ML Program format."""
        total_params = 0
        program = self._spec.mlProgram
        
        for func in program.functions.values():
            for block_name, block in func.block_specializations.items():
                for op in block.operations:
                    if op.type == "const":
                        for out in op.outputs:
                            if out.HasField("type"):
                                shape = self._type_to_shape(out.type)
                                if shape:
                                    elements = int(np.prod([d for d in shape if d > 0]))
                                    total_params += elements
        
        return total_params
    
    def _count_neural_network_params(self) -> int:
        """Count parameters in Neural Network format."""
        total_params = 0
        
        nn_spec = None
        if self._spec.HasField("neuralNetwork"):
            nn_spec = self._spec.neuralNetwork
        elif self._spec.HasField("neuralNetworkClassifier"):
            nn_spec = self._spec.neuralNetworkClassifier
        elif self._spec.HasField("neuralNetworkRegressor"):
            nn_spec = self._spec.neuralNetworkRegressor
        
        if nn_spec is None:
            return total_params
        
        for layer in nn_spec.layers:
            layer_type = layer.WhichOneof("layer")
            
            if layer_type == "convolution":
                conv = layer.convolution
                kernel_size = int(np.prod(conv.kernelSize)) if conv.kernelSize else 9
                total_params += conv.outputChannels * conv.kernelChannels * kernel_size
            
            elif layer_type == "innerProduct":
                ip = layer.innerProduct
                total_params += ip.inputChannels * ip.outputChannels + ip.outputChannels
        
        return total_params
    
    def _type_to_shape(self, type_spec) -> Optional[Tuple[int, ...]]:
        """Convert a type specification to a shape tuple."""
        if type_spec.HasField("tensorType"):
            dims = []
            for dim in type_spec.tensorType.dimensions:
                if dim.HasField("constant"):
                    dims.append(dim.constant.size)
                else:
                    dims.append(-1)
            return tuple(dims)
        return None
    
    def _type_to_dtype(self, type_spec) -> str:
        """Convert a type specification to a dtype string."""
        if type_spec.HasField("tensorType"):
            dtype_val = type_spec.tensorType.dataType
            dtype_map = {
                1: "float32",
                2: "float64",
                3: "int32",
                4: "int64",
                5: "float16",
                6: "int16",
                7: "int8",
                8: "uint8",
                9: "bool",
            }
            return dtype_map.get(dtype_val, "float32")
        return "float32"
