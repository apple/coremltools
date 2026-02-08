# Model Complexity Analyzer

A comprehensive tool for analyzing the computational complexity of Core ML models.

## Problem Statement

Before deploying Core ML models to Apple devices, developers need to understand:
- How much compute (FLOPS) a model requires
- How much memory the model will consume
- Which layers are the computational bottlenecks

**Previously, there was no tool in coremltools to answer these questions.**

This module fills that gap by providing:
1. FLOPS estimation for all MIL operations
2. Memory footprint analysis (parameters + activations)
3. Per-layer profiling
4. Human-readable reports in markdown and text formats

## Installation

This module is part of coremltools. No additional installation required.

## Quick Start

```python
import coremltools as ct
from coremltools.models.complexity import generate_report

model = ct.models.MLModel("my_model.mlpackage")
report = generate_report(model, "MyModel")

print(f"Total GFLOPS: {report.total_gflops:.2f}")
print(f"Memory: {report.memory_breakdown.total_mb:.2f} MB")
print(f"Parameters: {report.parameters_millions:.2f}M")

print(report.to_markdown())
```

## API Reference

### FLOPSAnalyzer

Analyze floating-point operations for a model.

```python
from coremltools.models.complexity import FLOPSAnalyzer

analyzer = FLOPSAnalyzer(model)
total_flops = analyzer.get_total_flops()
breakdown = analyzer.get_layer_breakdown()
by_type = analyzer.get_flops_by_op_type()
```

### MemoryEstimator

Estimate memory requirements.

```python
from coremltools.models.complexity import MemoryEstimator

estimator = MemoryEstimator(model)
breakdown = estimator.estimate()
param_count = estimator.get_parameter_count()

print(f"Total: {breakdown.total_mb:.2f} MB")
print(f"Parameters: {breakdown.parameter_mb:.2f} MB")
print(f"Activations: {breakdown.activation_mb:.2f} MB")
```

### LayerProfiler

Get detailed per-layer analysis.

```python
from coremltools.models.complexity import LayerProfiler

profiler = LayerProfiler(model)
profiles = profiler.profile()

top_layers = profiler.get_top_layers(n=5, by="flops")
for layer in top_layers:
    print(f"{layer.name}: {layer.mflops:.2f} MFLOPS")
```

### generate_report

Generate a complete report combining all analyses.

```python
from coremltools.models.complexity import generate_report

report = generate_report(model, "ModelName")

print(report.to_markdown())
print(report.to_text())
report_dict = report.to_dict()
```

## Supported Operations

FLOPS calculation is supported for:
- Convolutions (conv, conv_transpose)
- Linear/Fully-connected layers
- Matrix multiplications (matmul, einsum)
- Activations (relu, sigmoid, tanh, gelu, silu, softplus)
- Softmax
- Normalization (batch_norm, layer_norm, instance_norm)
- Pooling (max_pool, avg_pool)
- Element-wise operations (add, sub, mul, div)
- Reductions (reduce_sum, reduce_mean, reduce_max, reduce_min)

## Example Output

```
============================================================
Model Complexity Report: ResNet50
============================================================
Generated: 2025-02-09T00:45:00

SUMMARY
----------------------------------------
  Total FLOPS:            4,089,184,256
  Total GFLOPS:                    4.09
  Total MAC Ops:          2,044,592,128
  Parameters:                25,557,032
  Parameters (M):                 25.56

MEMORY ANALYSIS
----------------------------------------
  Total Memory:              106.42 MB
  Parameters:                 97.52 MB
  Activations:                 0.00 MB

TOP OPERATIONS BY FLOPS
----------------------------------------
  conv                 3,891,200,000 ( 95.2%)
  linear                 102,400,000 (  2.5%)
  batch_norm              51,200,000 (  1.3%)
  softmax                  1,000,000 (  0.0%)

============================================================
```

## Use Cases

1. **Model Optimization**: Identify bottleneck layers before applying compression
2. **Deployment Planning**: Estimate if a model fits device constraints
3. **Model Comparison**: Compare efficiency of different architectures
4. **Research**: Analyze compute/memory trade-offs

## Contributing

This module follows the coremltools contribution guidelines.
To add support for new operations, extend the `_calculate_flops_for_op_type` 
method in `flops_analyzer.py`.
