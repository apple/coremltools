# Copyright (c) 2025, Apple Inc. All rights reserved.
#
# Use of this source code is governed by a BSD-3-clause license that can be
# found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

"""
Tests for the Model Complexity Analyzer module.
"""

import pytest
import numpy as np
import tempfile
import os

import coremltools as ct
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import Program, Function

try:
    from coremltools.models.complexity import (
        FLOPSAnalyzer,
        LayerFLOPS,
        MemoryEstimator,
        MemoryBreakdown,
        LayerProfiler,
        LayerProfile,
        ComputeCostReport,
        generate_report,
    )
    HAS_COMPLEXITY_MODULE = True
except ImportError:
    HAS_COMPLEXITY_MODULE = False


import sys
import platform

SKIP_NATIVE_TESTS = sys.platform != "darwin"
SKIP_REASON = "coremltools native bindings only available on macOS"


def _create_simple_mlprogram():
    """Create a simple ML Program for testing."""
    if SKIP_NATIVE_TESTS:
        pytest.skip(SKIP_REASON)
    
    @mb.program(
        input_specs=[mb.TensorSpec(shape=(1, 3, 224, 224))],
        opset_version=ct.target.iOS15
    )
    def prog(x):
        x = mb.conv(x=x, weight=np.random.randn(32, 3, 3, 3).astype(np.float32), name="conv1")
        x = mb.relu(x=x, name="relu1")
        x = mb.reduce_mean(x=x, axes=[2, 3], keep_dims=False, name="pool")
        x = mb.linear(x=x, weight=np.random.randn(10, 32).astype(np.float32), name="fc")
        return x
    
    model = ct.convert(
        prog,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS15,
    )
    return model


def _create_matmul_model():
    """Create a model with matrix multiplication."""
    if SKIP_NATIVE_TESTS:
        pytest.skip(SKIP_REASON)
    
    @mb.program(
        input_specs=[mb.TensorSpec(shape=(1, 64, 128))],
        opset_version=ct.target.iOS15
    )
    def prog(x):
        weight = np.random.randn(128, 256).astype(np.float32)
        x = mb.matmul(x=x, y=weight, name="matmul1")
        return x
    
    model = ct.convert(
        prog,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS15,
    )
    return model


@pytest.mark.skipif(not HAS_COMPLEXITY_MODULE, reason="Complexity module not available")
class TestFLOPSAnalyzer:
    """Tests for FLOPSAnalyzer class."""
    
    def test_analyzer_creation(self):
        """Test that analyzer can be created with a model."""
        model = _create_simple_mlprogram()
        analyzer = FLOPSAnalyzer(model)
        assert analyzer is not None
    
    def test_analyze_returns_list(self):
        """Test that analyze returns a list of LayerFLOPS."""
        model = _create_simple_mlprogram()
        analyzer = FLOPSAnalyzer(model)
        result = analyzer.analyze()
        assert isinstance(result, list)
    
    def test_get_total_flops(self):
        """Test that total FLOPS is computed."""
        model = _create_simple_mlprogram()
        analyzer = FLOPSAnalyzer(model)
        total = analyzer.get_total_flops()
        assert isinstance(total, int)
        assert total > 0
    
    def test_get_total_mac_ops(self):
        """Test that total MAC operations are computed."""
        model = _create_simple_mlprogram()
        analyzer = FLOPSAnalyzer(model)
        total = analyzer.get_total_mac_ops()
        assert isinstance(total, int)
        assert total >= 0
    
    def test_get_layer_breakdown(self):
        """Test layer breakdown is sorted by FLOPS."""
        model = _create_simple_mlprogram()
        analyzer = FLOPSAnalyzer(model)
        breakdown = analyzer.get_layer_breakdown()
        
        if len(breakdown) >= 2:
            for i in range(len(breakdown) - 1):
                assert breakdown[i].flops >= breakdown[i + 1].flops
    
    def test_get_flops_by_op_type(self):
        """Test FLOPS aggregation by operation type."""
        model = _create_simple_mlprogram()
        analyzer = FLOPSAnalyzer(model)
        by_type = analyzer.get_flops_by_op_type()
        
        assert isinstance(by_type, dict)
        assert len(by_type) > 0
    
    def test_layer_flops_properties(self):
        """Test LayerFLOPS dataclass properties."""
        layer = LayerFLOPS(
            name="test",
            op_type="conv",
            flops=1_000_000_000,
            mac_ops=500_000_000,
            input_shapes=[(1, 3, 224, 224)],
            output_shapes=[(1, 64, 112, 112)],
        )
        
        assert layer.gflops == 1.0
        assert layer.mflops == 1000.0
    
    def test_matmul_flops(self):
        """Test FLOPS calculation for matmul."""
        model = _create_matmul_model()
        analyzer = FLOPSAnalyzer(model)
        total = analyzer.get_total_flops()
        assert total > 0


@pytest.mark.skipif(not HAS_COMPLEXITY_MODULE, reason="Complexity module not available")
class TestMemoryEstimator:
    """Tests for MemoryEstimator class."""
    
    def test_estimator_creation(self):
        """Test that estimator can be created."""
        model = _create_simple_mlprogram()
        estimator = MemoryEstimator(model)
        assert estimator is not None
    
    def test_estimate_returns_breakdown(self):
        """Test that estimate returns MemoryBreakdown."""
        model = _create_simple_mlprogram()
        estimator = MemoryEstimator(model)
        breakdown = estimator.estimate()
        
        assert isinstance(breakdown, MemoryBreakdown)
        assert breakdown.total_bytes > 0
    
    def test_get_parameter_count(self):
        """Test parameter counting."""
        model = _create_simple_mlprogram()
        estimator = MemoryEstimator(model)
        count = estimator.get_parameter_count()
        
        assert isinstance(count, int)
        assert count >= 0
    
    def test_memory_breakdown_properties(self):
        """Test MemoryBreakdown properties."""
        breakdown = MemoryBreakdown(
            total_bytes=104_857_600,
            parameter_bytes=83_886_080,
            activation_bytes=10_485_760,
            overhead_bytes=10_485_760,
        )
        
        assert breakdown.total_mb == 100.0
        assert breakdown.parameter_mb == 80.0
        assert breakdown.activation_mb == 10.0
    
    def test_memory_breakdown_to_dict(self):
        """Test MemoryBreakdown to_dict method."""
        breakdown = MemoryBreakdown(
            total_bytes=100,
            parameter_bytes=80,
            activation_bytes=10,
            overhead_bytes=10,
        )
        
        d = breakdown.to_dict()
        assert "total_bytes" in d
        assert "total_mb" in d
        assert "parameter_bytes" in d


@pytest.mark.skipif(not HAS_COMPLEXITY_MODULE, reason="Complexity module not available")
class TestLayerProfiler:
    """Tests for LayerProfiler class."""
    
    def test_profiler_creation(self):
        """Test that profiler can be created."""
        model = _create_simple_mlprogram()
        profiler = LayerProfiler(model)
        assert profiler is not None
    
    def test_profile_returns_list(self):
        """Test that profile returns list of LayerProfile."""
        model = _create_simple_mlprogram()
        profiler = LayerProfiler(model)
        profiles = profiler.profile()
        
        assert isinstance(profiles, list)
    
    def test_get_profiles_by_type(self):
        """Test grouping profiles by operation type."""
        model = _create_simple_mlprogram()
        profiler = LayerProfiler(model)
        by_type = profiler.get_profiles_by_type()
        
        assert isinstance(by_type, dict)
    
    def test_get_top_layers(self):
        """Test getting top layers by FLOPS."""
        model = _create_simple_mlprogram()
        profiler = LayerProfiler(model)
        top = profiler.get_top_layers(n=3, by="flops")
        
        assert isinstance(top, list)
        assert len(top) <= 3
    
    def test_get_summary(self):
        """Test getting profiling summary."""
        model = _create_simple_mlprogram()
        profiler = LayerProfiler(model)
        summary = profiler.get_summary()
        
        assert "num_layers" in summary
        assert "total_flops" in summary
        assert "total_memory_bytes" in summary
    
    def test_layer_profile_to_dict(self):
        """Test LayerProfile to_dict method."""
        profile = LayerProfile(
            name="test",
            op_type="conv",
            flops=1000,
            mac_ops=500,
            memory_bytes=4096,
            input_shapes=[(1, 3, 224, 224)],
            output_shapes=[(1, 64, 112, 112)],
        )
        
        d = profile.to_dict()
        assert d["name"] == "test"
        assert d["op_type"] == "conv"
        assert d["flops"] == 1000


@pytest.mark.skipif(not HAS_COMPLEXITY_MODULE, reason="Complexity module not available")
class TestComputeCostReport:
    """Tests for ComputeCostReport and generate_report."""
    
    def test_generate_report(self):
        """Test report generation."""
        model = _create_simple_mlprogram()
        report = generate_report(model, "TestModel")
        
        assert isinstance(report, ComputeCostReport)
        assert report.model_name == "TestModel"
        assert report.total_flops >= 0
    
    def test_report_to_markdown(self):
        """Test markdown generation."""
        model = _create_simple_mlprogram()
        report = generate_report(model, "TestModel")
        md = report.to_markdown()
        
        assert isinstance(md, str)
        assert "TestModel" in md
        assert "FLOPS" in md
    
    def test_report_to_text(self):
        """Test text report generation."""
        model = _create_simple_mlprogram()
        report = generate_report(model, "TestModel")
        text = report.to_text()
        
        assert isinstance(text, str)
        assert "TestModel" in text
    
    def test_report_to_dict(self):
        """Test dictionary export."""
        model = _create_simple_mlprogram()
        report = generate_report(model, "TestModel")
        d = report.to_dict()
        
        assert isinstance(d, dict)
        assert d["model_name"] == "TestModel"
        assert "summary" in d
        assert "memory" in d
    
    def test_report_properties(self):
        """Test report computed properties."""
        breakdown = MemoryBreakdown(
            total_bytes=100,
            parameter_bytes=80,
            activation_bytes=10,
            overhead_bytes=10,
        )
        
        report = ComputeCostReport(
            model_name="Test",
            total_flops=1_000_000_000,
            total_mac_ops=500_000_000,
            total_parameters=25_000_000,
            memory_breakdown=breakdown,
            layer_breakdown=[],
            flops_by_op_type={},
        )
        
        assert report.total_gflops == 1.0
        assert report.total_mflops == 1000.0
        assert report.parameters_millions == 25.0


@pytest.mark.skipif(not HAS_COMPLEXITY_MODULE, reason="Complexity module not available")
class TestModelPathInput:
    """Tests for loading models from file paths."""
    
    def test_analyzer_with_path(self):
        """Test FLOPSAnalyzer with model path."""
        model = _create_simple_mlprogram()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.mlpackage")
            model.save(path)
            
            analyzer = FLOPSAnalyzer(path)
            total = analyzer.get_total_flops()
            assert total > 0
    
    def test_estimator_with_path(self):
        """Test MemoryEstimator with model path."""
        model = _create_simple_mlprogram()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.mlpackage")
            model.save(path)
            
            estimator = MemoryEstimator(path)
            breakdown = estimator.estimate()
            assert breakdown.total_bytes > 0
    
    def test_profiler_with_path(self):
        """Test LayerProfiler with model path."""
        model = _create_simple_mlprogram()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.mlpackage")
            model.save(path)
            
            profiler = LayerProfiler(path)
            profiles = profiler.profile()
            assert isinstance(profiles, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
