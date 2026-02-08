# Copyright (c) 2025, Apple Inc. All rights reserved.
#
# Use of this source code is governed by a BSD-3-clause license that can be
# found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

"""
Compute Cost Report Generator for Core ML Models.

Generates human-readable reports summarizing model complexity metrics
including FLOPS, memory, and computational cost analysis.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime

from .flops_analyzer import FLOPSAnalyzer, LayerFLOPS
from .memory_estimator import MemoryEstimator, MemoryBreakdown


@dataclass
class ComputeCostReport:
    """Complete compute cost report for a Core ML model."""
    
    model_name: str
    total_flops: int
    total_mac_ops: int
    total_parameters: int
    memory_breakdown: MemoryBreakdown
    layer_breakdown: List[LayerFLOPS]
    flops_by_op_type: Dict[str, int]
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    @property
    def total_gflops(self) -> float:
        """Total FLOPS in billions."""
        return self.total_flops / 1e9
    
    @property
    def total_mflops(self) -> float:
        """Total FLOPS in millions."""
        return self.total_flops / 1e6
    
    @property
    def parameters_millions(self) -> float:
        """Parameters in millions."""
        return self.total_parameters / 1e6
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return {
            "model_name": self.model_name,
            "generated_at": self.generated_at,
            "summary": {
                "total_flops": self.total_flops,
                "total_gflops": self.total_gflops,
                "total_mac_ops": self.total_mac_ops,
                "total_parameters": self.total_parameters,
                "parameters_millions": self.parameters_millions,
            },
            "memory": self.memory_breakdown.to_dict(),
            "flops_by_op_type": self.flops_by_op_type,
            "top_layers": [
                {
                    "name": layer.name,
                    "op_type": layer.op_type,
                    "flops": layer.flops,
                    "mflops": layer.mflops,
                }
                for layer in self.layer_breakdown[:10]
            ],
        }
    
    def to_markdown(self) -> str:
        """Generate a markdown formatted report."""
        lines = [
            f"# Model Complexity Report: {self.model_name}",
            "",
            f"*Generated at: {self.generated_at}*",
            "",
            "## Summary",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Total FLOPS | {self.total_flops:,} |",
            f"| Total GFLOPS | {self.total_gflops:.2f} |",
            f"| Total MAC Ops | {self.total_mac_ops:,} |",
            f"| Total Parameters | {self.total_parameters:,} |",
            f"| Parameters (M) | {self.parameters_millions:.2f} |",
            "",
            "## Memory Analysis",
            "",
            "| Component | Size |",
            "|-----------|------|",
            f"| Total Memory | {self.memory_breakdown.total_mb:.2f} MB |",
            f"| Parameter Memory | {self.memory_breakdown.parameter_mb:.2f} MB |",
            f"| Activation Memory | {self.memory_breakdown.activation_mb:.2f} MB |",
            f"| Overhead | {self.memory_breakdown.overhead_bytes / (1024*1024):.2f} MB |",
            "",
            "## FLOPS by Operation Type",
            "",
            "| Operation | FLOPS | Percentage |",
            "|-----------|-------|------------|",
        ]
        
        for op_type, flops in list(self.flops_by_op_type.items())[:15]:
            percentage = (flops / self.total_flops * 100) if self.total_flops > 0 else 0
            lines.append(f"| {op_type} | {flops:,} | {percentage:.1f}% |")
        
        lines.extend([
            "",
            "## Top 10 Layers by FLOPS",
            "",
            "| Layer | Type | FLOPS | MFLOPS |",
            "|-------|------|-------|--------|",
        ])
        
        for layer in self.layer_breakdown[:10]:
            lines.append(
                f"| {layer.name[:40]} | {layer.op_type} | {layer.flops:,} | {layer.mflops:.2f} |"
            )
        
        return "\n".join(lines)
    
    def to_text(self) -> str:
        """Generate a plain text report."""
        separator = "=" * 60
        lines = [
            separator,
            f"Model Complexity Report: {self.model_name}",
            separator,
            f"Generated: {self.generated_at}",
            "",
            "SUMMARY",
            "-" * 40,
            f"  Total FLOPS:      {self.total_flops:>15,}",
            f"  Total GFLOPS:     {self.total_gflops:>15.2f}",
            f"  Total MAC Ops:    {self.total_mac_ops:>15,}",
            f"  Parameters:       {self.total_parameters:>15,}",
            f"  Parameters (M):   {self.parameters_millions:>15.2f}",
            "",
            "MEMORY ANALYSIS",
            "-" * 40,
            f"  Total Memory:     {self.memory_breakdown.total_mb:>12.2f} MB",
            f"  Parameters:       {self.memory_breakdown.parameter_mb:>12.2f} MB",
            f"  Activations:      {self.memory_breakdown.activation_mb:>12.2f} MB",
            "",
            "TOP OPERATIONS BY FLOPS",
            "-" * 40,
        ]
        
        for op_type, flops in list(self.flops_by_op_type.items())[:10]:
            percentage = (flops / self.total_flops * 100) if self.total_flops > 0 else 0
            lines.append(f"  {op_type:<20} {flops:>12,} ({percentage:>5.1f}%)")
        
        lines.extend([
            "",
            separator,
        ])
        
        return "\n".join(lines)


def generate_report(model, model_name: Optional[str] = None) -> ComputeCostReport:
    """
    Generate a comprehensive compute cost report for a Core ML model.
    
    Parameters
    ----------
    model : MLModel or str
        A loaded MLModel instance or path to a .mlmodel/.mlpackage file.
    model_name : str, optional
        Name to use in the report. If not provided, uses "Unnamed Model".
    
    Returns
    -------
    ComputeCostReport
        Complete report with FLOPS, memory, and layer breakdown.
    
    Examples
    --------
    >>> import coremltools as ct
    >>> from coremltools.models.complexity import generate_report
    >>> 
    >>> model = ct.models.MLModel("my_model.mlpackage")
    >>> report = generate_report(model, "MyModel")
    >>> print(report.to_markdown())
    >>> 
    >>> print(f"Total GFLOPS: {report.total_gflops:.2f}")
    >>> print(f"Memory: {report.memory_breakdown.total_mb:.2f} MB")
    """
    if model_name is None:
        model_name = "Unnamed Model"
    
    flops_analyzer = FLOPSAnalyzer(model)
    memory_estimator = MemoryEstimator(model)
    
    layer_breakdown = flops_analyzer.analyze()
    flops_by_op_type = flops_analyzer.get_flops_by_op_type()
    memory_breakdown = memory_estimator.estimate()
    
    return ComputeCostReport(
        model_name=model_name,
        total_flops=flops_analyzer.get_total_flops(),
        total_mac_ops=flops_analyzer.get_total_mac_ops(),
        total_parameters=memory_estimator.get_parameter_count(),
        memory_breakdown=memory_breakdown,
        layer_breakdown=layer_breakdown,
        flops_by_op_type=flops_by_op_type,
    )
