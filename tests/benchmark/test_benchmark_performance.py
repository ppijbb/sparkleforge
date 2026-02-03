"""
Performance Benchmark Tests

Tests for performance metrics including response time measurement,
throughput testing, resource usage tracking, and real-time update latency.
"""

import pytest
import time
import logging
from typing import Dict, List, Any
from pathlib import Path

from .benchmark_runner import BenchmarkRunner
from .benchmark_metrics import MetricsCollector, BenchmarkResult

logger = logging.getLogger(__name__)


class TestPerformanceBenchmark:
    """Test performance benchmark functionality."""
    
    @pytest.fixture
    def benchmark_runner(self, project_root):
        """Create benchmark runner instance."""
        config_path = Path(project_root) / "tests" / "benchmark_config.yaml"
        thresholds_path = Path(project_root) / "tests" / "benchmark_thresholds.yaml"
        return BenchmarkRunner(str(project_root), str(config_path), str(thresholds_path))
    
    @pytest.fixture
    def metrics_collector(self, benchmark_runner):
        """Create metrics collector instance."""
        return benchmark_runner.metrics_collector
    
    def test_response_time_measurement(self, metrics_collector):
        """Test response time measurement."""
        start_time = time.time()
        time.sleep(0.1)  # Simulate some processing time
        end_time = time.time()
        
        result = metrics_collector.measure_response_time(start_time, end_time)
        
        assert result.name == "response_time"
        assert result.value > 0.09  # Should be close to 0.1 seconds
        assert result.value < 0.2   # But not too much more
        assert result.category == "performance"
        assert "start_time" in result.metadata
        assert "end_time" in result.metadata
        assert "formatted_time" in result.metadata
    
    def test_response_time_threshold_evaluation(self, metrics_collector):
        """Test response time threshold evaluation."""
        # Test with good response time
        start_time = time.time()
        end_time = start_time + 60.0  # 60 seconds
        
        result = metrics_collector.measure_response_time(start_time, end_time)
        
        assert result.value == 60.0
        assert result.passed  # Should pass default threshold of 120s
        
        # Test with poor response time
        end_time = start_time + 150.0  # 150 seconds
        
        result = metrics_collector.measure_response_time(start_time, end_time)
        
        assert result.value == 150.0
        assert not result.passed  # Should fail threshold of 120s
    
    def test_performance_benchmark_execution(self, benchmark_runner):
        """Test that performance benchmark can be executed."""
        results = benchmark_runner.run_performance_benchmark()
        
        assert isinstance(results, list)
        assert len(results) > 0, "Should have at least one test result"
        
        # Check that all results have execution time
        for result in results:
            assert isinstance(result, BenchmarkResult)
            assert result.execution_time >= 0
            assert result.test_id is not None
    
    def test_performance_metrics_collection(self, benchmark_runner):
        """Test that performance metrics are properly collected."""
        test_case = {
            "id": "test-perf-001",
            "category": "Technology",
            "query": "AI performance",
            "expected": {
                "max_response_time": 120
            }
        }
        
        extracted_metrics = {
            "sources": [],
            "execution_results": []
        }
        
        metrics = benchmark_runner._collect_metrics_for_test(test_case, extracted_metrics, 60.0)
        
        # Should have response time metric
        metric_names = [m.name for m in metrics]
        assert "response_time" in metric_names
        
        # Check response time metric
        response_time_metric = next(m for m in metrics if m.name == "response_time")
        assert response_time_metric.category == "performance"
        assert response_time_metric.value == 60.0
    
    def test_performance_threshold_evaluation(self, benchmark_runner):
        """Test performance threshold evaluation."""
        test_case = {
            "id": "test-perf-threshold",
            "category": "Business",
            "query": "Business performance",
            "expected": {
                "max_response_time": 100  # 100 seconds threshold
            }
        }
        
        # Create metrics that should pass
        metrics = [
            type('Metric', (), {
                'name': 'response_time',
                'value': 80.0,
                'threshold': 100.0,
                'passed': True,
                'category': 'performance'
            })()
        ]
        
        passed = benchmark_runner._evaluate_test_pass(metrics, test_case['expected'])
        assert passed, "Test should pass with good response time"
        
        # Create metrics that should fail
        failing_metrics = [
            type('Metric', (), {
                'name': 'response_time',
                'value': 120.0,
                'threshold': 100.0,
                'passed': False,
                'category': 'performance'
            })()
        ]
        
        failed = benchmark_runner._evaluate_test_pass(failing_metrics, test_case['expected'])
        assert not failed, "Test should fail with poor response time"
    
    def test_throughput_calculation(self, benchmark_runner):
        """Test throughput calculation from multiple results."""
        # Create mock results with different execution times
        mock_results = [
            BenchmarkResult(
                test_id="test-001",
                category="Technology",
                query="Query 1",
                execution_time=60.0,  # 1 minute
                metrics=[],
                overall_score=0.8,
                passed=True,
                timestamp=None
            ),
            BenchmarkResult(
                test_id="test-002",
                category="Technology",
                query="Query 2",
                execution_time=120.0,  # 2 minutes
                metrics=[],
                overall_score=0.7,
                passed=True,
                timestamp=None
            ),
            BenchmarkResult(
                test_id="test-003",
                category="Technology",
                query="Query 3",
                execution_time=90.0,  # 1.5 minutes
                metrics=[],
                overall_score=0.9,
                passed=True,
                timestamp=None
            )
        ]
        
        summary = benchmark_runner.get_benchmark_summary(mock_results)
        
        assert summary["total_tests"] == 3
        assert summary["average_execution_time"] == 90.0  # (60 + 120 + 90) / 3
        
        # Calculate throughput (queries per minute)
        total_time_minutes = sum(r.execution_time for r in mock_results) / 60
        throughput = len(mock_results) / total_time_minutes
        assert throughput > 0
    
    def test_performance_benchmark_summary(self, benchmark_runner):
        """Test performance benchmark summary generation."""
        mock_results = [
            BenchmarkResult(
                test_id="perf-001",
                category="Technology",
                query="Fast query",
                execution_time=30.0,
                metrics=[],
                overall_score=0.9,
                passed=True,
                timestamp=None
            ),
            BenchmarkResult(
                test_id="perf-002",
                category="Business",
                query="Slow query",
                execution_time=150.0,
                metrics=[],
                overall_score=0.6,
                passed=False,
                timestamp=None
            )
        ]
        
        summary = benchmark_runner.get_benchmark_summary(mock_results)
        
        assert summary["total_tests"] == 2
        assert summary["passed_tests"] == 1
        assert summary["failed_tests"] == 1
        assert summary["pass_rate"] == 0.5
        assert summary["average_execution_time"] == 90.0  # (30 + 150) / 2
        
        # Check category breakdown
        assert "category_breakdown" in summary
        breakdown = summary["category_breakdown"]
        assert "Technology" in breakdown
        assert "Business" in breakdown
        assert breakdown["Technology"]["total"] == 1
        assert breakdown["Business"]["total"] == 1
    
    def test_concurrent_execution_simulation(self, benchmark_runner):
        """Test simulation of concurrent execution."""
        # This tests the parallel execution logic without actually running CLI
        test_cases = [
            {"id": "concurrent-001", "category": "Technology", "query": "Query 1"},
            {"id": "concurrent-002", "category": "Business", "query": "Query 2"},
            {"id": "concurrent-003", "category": "Technology", "query": "Query 3"}
        ]
        
        # Simulate parallel execution by running sequentially but measuring time
        start_time = time.time()
        results = benchmark_runner._run_sequential_benchmarks(test_cases)
        total_time = time.time() - start_time
        
        assert len(results) == 3
        assert total_time >= 0  # Should complete in reasonable time
    
    def test_execution_time_accuracy(self, metrics_collector):
        """Test that execution time measurement is accurate."""
        # Test with known time difference
        start_time = time.time()
        time.sleep(0.05)  # 50ms
        end_time = time.time()
        
        result = metrics_collector.measure_response_time(start_time, end_time)
        
        # Should be close to 0.05 seconds
        assert 0.04 <= result.value <= 0.06
        assert result.passed  # Should pass default threshold
    
    def test_performance_metrics_metadata(self, metrics_collector):
        """Test that performance metrics include proper metadata."""
        start_time = time.time()
        end_time = start_time + 45.0
        
        result = metrics_collector.measure_response_time(start_time, end_time)
        
        assert "start_time" in result.metadata
        assert "end_time" in result.metadata
        assert "formatted_time" in result.metadata
        assert result.metadata["formatted_time"] == "45.00s"
    
    @pytest.mark.slow
    def test_full_performance_benchmark(self, benchmark_runner):
        """Test full performance benchmark execution (slow test)."""
        # This test actually runs the CLI and may take time
        results = benchmark_runner.run_performance_benchmark()
        
        assert len(results) > 0, "Should have results from performance benchmark"
        
        # Check that we have results from performance-focused categories
        categories = {result.category for result in results}
        expected_categories = {"Technology", "Business"}
        assert categories.intersection(expected_categories), "Should have results from performance categories"
        
        # Check execution times are reasonable
        for result in results:
            assert result.execution_time > 0, "Should have positive execution time"
            assert result.execution_time < 600, "Should complete within 10 minutes"
    
    def test_performance_regression_detection(self, benchmark_runner):
        """Test performance regression detection."""
        # Create current results (slower)
        current_results = [
            BenchmarkResult(
                test_id="test-001",
                category="Technology",
                query="Query",
                execution_time=150.0,  # 150 seconds
                metrics=[],
                overall_score=0.8,
                passed=True,
                timestamp=None
            )
        ]
        
        # Create previous results (faster)
        previous_results = [
            BenchmarkResult(
                test_id="test-001",
                category="Technology",
                query="Query",
                execution_time=100.0,  # 100 seconds
                metrics=[],
                overall_score=0.8,
                passed=True,
                timestamp=None
            )
        ]
        
        regressions = benchmark_runner.analyzer.detect_regressions(current_results, previous_results)
        
        assert len(regressions) > 0, "Should detect performance regression"
        regression = regressions[0]
        assert regression["type"] == "performance_degradation"
        assert regression["degradation_percent"] > 0  # Should show degradation
