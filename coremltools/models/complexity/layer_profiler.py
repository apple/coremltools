# Copyright (c) 2025, Apple Inc. All rights reserved.
#
# Use of this source code is governed by a BSD-3-clause license that can be
# found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

"""
Layer Profiler for Core ML Models.

Provides detailed per-layer profiling capability for Core ML models,
enabling fine-grained analysis of individual operations.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from coremltools.models import MLModel

from .flops_analyzer import FLOPSAnalyzer, LayerFLOPS
from .memory_estimator import MemoryEstimator, DTYPE_BYTES


@dataclass
class LayerProfile:
    """Complete profile for a single layer."""
    
    name: str
    op_type: str
    flops: int
    mac_ops: int
    memory_bytes: int
    input_shapes: List[Tuple[int, ...]]
    output_shapes: List[Tuple[int, ...]]
    input_dtypes: List[str] = field(default_factory=list)
    output_dtypes: List[str] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def flops_percentage(self) -> float:
        """Placeholder for percentage - set by profiler."""
        return 0.0
    
    @property
    def memory_mb(self) -> float:
        """Memory in megabytes."""
        return self.memory_bytes / (1024 * 1024)
    
    @property
    def mflops(self) -> float:
        """FLOPS in millions."""
        return self.flops / 1e6
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "op_type": self.op_type,
            "flops": self.flops,
            "mflops": self.mflops,
            "mac_ops": self.mac_ops,
            "memory_bytes": self.memory_bytes,
            "memory_mb": self.memory_mb,
            "input_shapes": self.input_shapes,
            "output_shapes": self.output_shapes,
            "input_dtypes": self.input_dtypes,
            "output_dtypes": self.output_dtypes,
            "attributes": self.attributes,
        }


class LayerProfiler:
    """
    Profiler for detailed layer-by-layer analysis of Core ML models.
    
    Combines FLOPS, memory, and shape information for each layer
    to provide comprehensive profiling data.
    
    Parameters
    ----------
    model : MLModel or str
        A loaded MLModel instance or path to a .mlmodel/.mlpackage file.
    
    Examples
    --------
    >>> import coremltools as ct
    >>> from coremltools.models.complexity import LayerProfiler
    >>> 
    >>> model = ct.models.MLModel("my_model.mlpackage")
    >>> profiler = LayerProfiler(model)
    >>> profiles = profiler.profile()
    >>> 
    >>> for layer in profiles[:5]:
    ...     print(f"{layer.name}: {layer.mflops:.2f} MFLOPS, {layer.memory_mb:.2f} MB")
    """
    
    def __init__(self, model):
        if isinstance(model, str):
            self._model = MLModel(model)
        else:
            self._model = model
        
        self._spec = self._model.get_spec()
        self._profiles: List[LayerProfile] = []
        self._profiled = False
    
    def profile(self) -> List[LayerProfile]:
        """
        Profile all layers in the model.
        
        Returns
        -------
        List[LayerProfile]
            List of layer profiles.
        """
        self._profiles = []
        
        flops_analyzer = FLOPSAnalyzer(self._model)
        layer_flops = flops_analyzer.analyze()
        
        flops_dict = {lf.name: lf for lf in layer_flops}
        
        if self._spec.WhichOneof("Type") == "mlProgram":
            self._profile_ml_program(flops_dict)
        elif self._spec.WhichOneof("Type") in ("neuralNetwork", "neuralNetworkClassifier", 
                                                 "neuralNetworkRegressor"):
            self._profile_neural_network(flops_dict)
        
        self._profiled = True
        return self._profiles
    
    def get_profiles_by_type(self) -> Dict[str, List[LayerProfile]]:
        """
        Get layer profiles grouped by operation type.
        
        Returns
        -------
        Dict[str, List[LayerProfile]]
            Dictionary mapping op type to list of profiles.
        """
        if not self._profiled:
            self.profile()
        
        by_type: Dict[str, List[LayerProfile]] = {}
        for profile in self._profiles:
            if profile.op_type not in by_type:
                by_type[profile.op_type] = []
            by_type[profile.op_type].append(profile)
        
        return by_type
    
    def get_top_layers(self, n: int = 10, by: str = "flops") -> List[LayerProfile]:
        """
        Get the top N layers by a specified metric.
        
        Parameters
        ----------
        n : int
            Number of layers to return.
        by : str
            Metric to sort by: "flops", "memory", or "mac".
        
        Returns
        -------
        List[LayerProfile]
            Top N layers sorted by the specified metric.
        """
        if not self._profiled:
            self.profile()
        
        key_map = {
            "flops": lambda x: x.flops,
            "memory": lambda x: x.memory_bytes,
            "mac": lambda x: x.mac_ops,
        }
        
        key_fn = key_map.get(by, key_map["flops"])
        sorted_profiles = sorted(self._profiles, key=key_fn, reverse=True)
        
        return sorted_profiles[:n]
    
    def find_layer(self, name: str) -> Optional[LayerProfile]:
        """
        Find a specific layer by name.
        
        Parameters
        ----------
        name : str
            Layer name to search for.
        
        Returns
        -------
        LayerProfile or None
            The layer profile if found, None otherwise.
        """
        if not self._profiled:
            self.profile()
        
        for profile in self._profiles:
            if profile.name == name:
                return profile
        return None
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the profiling results.
        
        Returns
        -------
        Dict[str, Any]
            Summary statistics.
        """
        if not self._profiled:
            self.profile()
        
        total_flops = sum(p.flops for p in self._profiles)
        total_mac = sum(p.mac_ops for p in self._profiles)
        total_memory = sum(p.memory_bytes for p in self._profiles)
        
        return {
            "num_layers": len(self._profiles),
            "total_flops": total_flops,
            "total_mac_ops": total_mac,
            "total_memory_bytes": total_memory,
            "total_memory_mb": total_memory / (1024 * 1024),
            "avg_flops_per_layer": total_flops / len(self._profiles) if self._profiles else 0,
            "unique_op_types": len(set(p.op_type for p in self._profiles)),
        }
    
    def _profile_ml_program(self, flops_dict: Dict[str, LayerFLOPS]):
        """Profile ML Program format models."""
        program = self._spec.mlProgram
        
        for func in program.functions.values():
            for block_name, block in func.block_specializations.items():
                for op in block.operations:
                    op_name = op.outputs[0].name if op.outputs else "unknown"
                    
                    layer_flops = flops_dict.get(op_name)
                    
                    input_shapes = self._extract_input_shapes(op)
                    output_shapes = self._extract_output_shapes(op)
                    input_dtypes = self._extract_input_dtypes(op)
                    output_dtypes = self._extract_output_dtypes(op)
                    
                    memory_bytes = self._estimate_layer_memory(output_shapes, output_dtypes)
                    
                    flops = layer_flops.flops if layer_flops else 0
                    mac_ops = layer_flops.mac_ops if layer_flops else 0
                    
                    profile = LayerProfile(
                        name=op_name,
                        op_type=op.type,
                        flops=flops,
                        mac_ops=mac_ops,
                        memory_bytes=memory_bytes,
                        input_shapes=input_shapes,
                        output_shapes=output_shapes,
                        input_dtypes=input_dtypes,
                        output_dtypes=output_dtypes,
                        attributes=self._extract_attributes(op),
                    )
                    
                    self._profiles.append(profile)
    
    def _profile_neural_network(self, flops_dict: Dict[str, LayerFLOPS]):
        """Profile Neural Network format models."""
        nn_spec = None
        if self._spec.HasField("neuralNetwork"):
            nn_spec = self._spec.neuralNetwork
        elif self._spec.HasField("neuralNetworkClassifier"):
            nn_spec = self._spec.neuralNetworkClassifier
        elif self._spec.HasField("neuralNetworkRegressor"):
            nn_spec = self._spec.neuralNetworkRegressor
        
        if nn_spec is None:
            return
        
        for layer in nn_spec.layers:
            layer_flops = flops_dict.get(layer.name)
            
            profile = LayerProfile(
                name=layer.name,
                op_type=layer.WhichOneof("layer"),
                flops=layer_flops.flops if layer_flops else 0,
                mac_ops=layer_flops.mac_ops if layer_flops else 0,
                memory_bytes=4096,
                input_shapes=[],
                output_shapes=[],
            )
            
            self._profiles.append(profile)
    
    def _extract_input_shapes(self, op) -> List[Tuple[int, ...]]:
        """Extract input shapes from an operation."""
        shapes = []
        for inp in op.inputs.values():
            if inp.arguments:
                arg = inp.arguments[0]
                if arg.HasField("type"):
                    shape = self._type_to_shape(arg.type)
                    if shape:
                        shapes.append(shape)
        return shapes
    
    def _extract_output_shapes(self, op) -> List[Tuple[int, ...]]:
        """Extract output shapes from an operation."""
        shapes = []
        for out in op.outputs:
            if out.HasField("type"):
                shape = self._type_to_shape(out.type)
                if shape:
                    shapes.append(shape)
        return shapes
    
    def _extract_input_dtypes(self, op) -> List[str]:
        """Extract input dtypes from an operation."""
        dtypes = []
        for inp in op.inputs.values():
            if inp.arguments:
                arg = inp.arguments[0]
                if arg.HasField("type"):
                    dtype = self._type_to_dtype(arg.type)
                    dtypes.append(dtype)
        return dtypes
    
    def _extract_output_dtypes(self, op) -> List[str]:
        """Extract output dtypes from an operation."""
        dtypes = []
        for out in op.outputs:
            if out.HasField("type"):
                dtype = self._type_to_dtype(out.type)
                dtypes.append(dtype)
        return dtypes
    
    def _estimate_layer_memory(
        self, 
        shapes: List[Tuple[int, ...]], 
        dtypes: List[str]
    ) -> int:
        """Estimate memory for a layer based on output shapes."""
        total_bytes = 0
        
        for i, shape in enumerate(shapes):
            if not shape:
                continue
            
            dtype = dtypes[i] if i < len(dtypes) else "float32"
            bytes_per_element = DTYPE_BYTES.get(dtype, 4)
            
            elements = int(np.prod([d for d in shape if d > 0]))
            total_bytes += elements * bytes_per_element
        
        return total_bytes
    
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
    
    def _extract_attributes(self, op) -> Dict[str, Any]:
        """Extract attributes from an operation."""
        attrs = {}
        for attr_name, attr_val in op.attributes.items():
            if attr_val.HasField("i"):
                attrs[attr_name] = attr_val.i
            elif attr_val.HasField("f"):
                attrs[attr_name] = attr_val.f
            elif attr_val.HasField("b"):
                attrs[attr_name] = attr_val.b
            elif attr_val.HasField("s"):
                attrs[attr_name] = attr_val.s
        return attrs
