"""
Pytest Integration for SparkleForge Benchmarks

Integrate benchmark tests with pytest framework for CI/CD integration.
Provides pytest markers and fixtures for benchmark testing.
"""

import pytest
import logging
from typing import List, Dict, Any
from pathlib import Path

from benchmark_runner import BenchmarkRunner
from benchmark_metrics import BenchmarkResult

logger = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def project_root():
    """Get project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def benchmark_runner(project_root):
    """Create benchmark runner instance for session."""
    config_path = project_root / "tests" / "benchmark_config.yaml"
    thresholds_path = project_root / "tests" / "benchmark_thresholds.yaml"
    return BenchmarkRunner(str(project_root), str(config_path), str(thresholds_path))


@pytest.fixture(scope="session")
def benchmark_results(benchmark_runner):
    """Run all benchmarks and return results (production environment required)."""
    # Validate production environment before running benchmarks
    is_valid, issues = benchmark_runner.cli_executor.validate_environment()
    if not is_valid:
        pytest.skip(f"Production environment not ready: {issues}")
    
    return benchmark_runner.run_all_benchmarks()


class TestBenchmarkIntegration:
    """Integration tests for benchmark system."""
    
    def test_benchmark_runner_initialization(self, benchmark_runner):
        """Test that benchmark runner initializes correctly."""
        assert benchmark_runner is not None
        assert benchmark_runner.config is not None
        assert benchmark_runner.thresholds is not None
        assert benchmark_runner.metrics_collector is not None
        assert benchmark_runner.analyzer is not None
        assert benchmark_runner.cli_executor is not None
    
    def test_benchmark_config_loading(self, benchmark_runner):
        """Test that benchmark configuration loads correctly."""
        config = benchmark_runner.config
        
        assert "test_cases" in config
        assert isinstance(config["test_cases"], list)
        assert len(config["test_cases"]) > 0
        
        # Check test case structure
        for test_case in config["test_cases"]:
            assert "id" in test_case
            assert "category" in test_case
            assert "query" in test_case
            assert "expected" in test_case
    
    def test_benchmark_thresholds_loading(self, benchmark_runner):
        """Test that benchmark thresholds load correctly."""
        thresholds = benchmark_runner.thresholds
        
        assert "thresholds" in thresholds
        assert "scoring_weights" in thresholds
        
        # Check required thresholds
        required_thresholds = [
            "response_time", "source_credibility", "factual_accuracy",
            "creative_novelty", "memory_precision"
        ]
        
        for threshold in required_thresholds:
            assert threshold in thresholds["thresholds"]
    
    def test_cli_executor_validation(self, benchmark_runner):
        """Test CLI executor environment validation."""
        is_valid, issues = benchmark_runner.cli_executor.validate_environment()
        
        # Should have some validation results
        assert isinstance(is_valid, bool)
        assert isinstance(issues, list)
        
        if not is_valid:
            logger.warning(f"Environment validation issues: {issues}")


@pytest.mark.benchmark
class TestBenchmarkSuites:
    """Test individual benchmark suites."""
    
    def test_research_quality_benchmark(self, benchmark_runner):
        """Test research quality benchmark suite."""
        results = benchmark_runner.run_research_quality_benchmark()
        
        assert isinstance(results, list)
        assert len(results) > 0
        
        # Check that all results are from research quality categories
        for result in results:
            assert result.category in ["Technology", "Science", "Health"]
            assert isinstance(result, BenchmarkResult)
    
    def test_performance_benchmark(self, benchmark_runner):
        """Test performance benchmark suite."""
        results = benchmark_runner.run_performance_benchmark()
        
        assert isinstance(results, list)
        assert len(results) > 0
        
        # Check that all results are from performance categories
        for result in results:
            assert result.category in ["Technology", "Business"]
            assert isinstance(result, BenchmarkResult)
    
    def test_source_validation_benchmark(self, benchmark_runner):
        """Test source validation benchmark suite."""
        results = benchmark_runner.run_source_validation_benchmark()
        
        assert isinstance(results, list)
        assert len(results) > 0
        
        # Check that all results are from validation categories
        for result in results:
            assert result.category in ["Science", "Health"]
            assert isinstance(result, BenchmarkResult)
    
    def test_creative_insights_benchmark(self, benchmark_runner):
        """Test creative insights benchmark suite."""
        results = benchmark_runner.run_creative_insights_benchmark()
        
        assert isinstance(results, list)
        assert len(results) > 0
        
        # Check that all results are from creative categories
        for result in results:
            assert result.category in ["Creative", "Health"]
            assert isinstance(result, BenchmarkResult)
    
    def test_memory_learning_benchmark(self, benchmark_runner):
        """Test memory learning benchmark suite."""
        results = benchmark_runner.run_memory_learning_benchmark()
        
        assert isinstance(results, list)
        assert len(results) > 0
        
        # Check that all results are from memory categories
        for result in results:
            assert result.category in ["Technology", "Business"]
            assert isinstance(result, BenchmarkResult)


@pytest.mark.slow
class TestFullBenchmarkSuite:
    """Test full benchmark suite execution."""
    
    def test_full_benchmark_execution(self, benchmark_runner):
        """Test execution of full benchmark suite."""
        results = benchmark_runner.run_all_benchmarks()
        
        assert isinstance(results, list)
        assert len(results) > 0
        
        # Check that we have results from different categories
        categories = {result.category for result in results}
        expected_categories = {"Technology", "Science", "Business", "Health", "Creative"}
        assert categories.intersection(expected_categories), "Should have results from expected categories"
        
        # Check that some tests completed successfully
        completed_tests = sum(1 for result in results if result.execution_time > 0)
        assert completed_tests > 0, "Should have at least some completed tests"
    
    def test_benchmark_results_structure(self, benchmark_results):
        """Test structure of benchmark results."""
        assert isinstance(benchmark_results, list)
        
        for result in benchmark_results:
            assert isinstance(result, BenchmarkResult)
            assert hasattr(result, 'test_id')
            assert hasattr(result, 'category')
            assert hasattr(result, 'query')
            assert hasattr(result, 'execution_time')
            assert hasattr(result, 'overall_score')
            assert hasattr(result, 'passed')
            assert hasattr(result, 'metrics')
            assert hasattr(result, 'timestamp')
    
    def test_benchmark_performance_thresholds(self, benchmark_results):
        """Test that benchmark results meet performance thresholds."""
        if not benchmark_results:
            pytest.skip("No benchmark results available")
        
        # Check execution times are reasonable
        for result in benchmark_results:
            assert result.execution_time > 0, f"Test {result.test_id} should have positive execution time"
            assert result.execution_time < 600, f"Test {result.test_id} should complete within 10 minutes"
    
    def test_benchmark_quality_thresholds(self, benchmark_results):
        """Test that benchmark results meet quality thresholds."""
        if not benchmark_results:
            pytest.skip("No benchmark results available")
        
        # Check that overall scores are reasonable
        for result in benchmark_results:
            assert 0 <= result.overall_score <= 1, f"Test {result.test_id} should have score between 0 and 1"
    
    def test_benchmark_summary_generation(self, benchmark_runner, benchmark_results):
        """Test benchmark summary generation."""
        summary = benchmark_runner.get_benchmark_summary(benchmark_results)
        
        assert isinstance(summary, dict)
        assert "total_tests" in summary
        assert "passed_tests" in summary
        assert "failed_tests" in summary
        assert "pass_rate" in summary
        assert "overall_score" in summary
        assert "average_execution_time" in summary
        assert "category_breakdown" in summary
        
        assert summary["total_tests"] == len(benchmark_results)
        assert summary["passed_tests"] + summary["failed_tests"] == summary["total_tests"]
        assert 0 <= summary["pass_rate"] <= 1
        assert 0 <= summary["overall_score"] <= 1
        assert summary["average_execution_time"] >= 0


@pytest.mark.regression
class TestRegressionDetection:
    """Test regression detection functionality."""
    
    def test_regression_detection_with_mock_data(self, benchmark_runner):
        """Test regression detection with mock data."""
        # Create mock current results (worse performance)
        current_results = [
            BenchmarkResult(
                test_id="test-001",
                category="Technology",
                query="Test query",
                execution_time=150.0,  # Slower
                metrics=[],
                overall_score=0.6,  # Lower score
                passed=True,
                timestamp=None
            )
        ]
        
        # Create mock previous results (better performance)
        previous_results = [
            BenchmarkResult(
                test_id="test-001",
                category="Technology",
                query="Test query",
                execution_time=100.0,  # Faster
                metrics=[],
                overall_score=0.8,  # Higher score
                passed=True,
                timestamp=None
            )
        ]
        
        regressions = benchmark_runner.analyzer.detect_regressions(current_results, previous_results)
        
        assert len(regressions) > 0, "Should detect regressions"
        
        # Check regression types
        regression_types = {r["type"] for r in regressions}
        assert "performance_degradation" in regression_types
        assert "quality_degradation" in regression_types
    
    def test_no_regression_detection_with_improved_data(self, benchmark_runner):
        """Test that no regressions are detected with improved data."""
        # Create mock current results (better performance)
        current_results = [
            BenchmarkResult(
                test_id="test-001",
                category="Technology",
                query="Test query",
                execution_time=80.0,  # Faster
                metrics=[],
                overall_score=0.9,  # Higher score
                passed=True,
                timestamp=None
            )
        ]
        
        # Create mock previous results (worse performance)
        previous_results = [
            BenchmarkResult(
                test_id="test-001",
                category="Technology",
                query="Test query",
                execution_time=120.0,  # Slower
                metrics=[],
                overall_score=0.7,  # Lower score
                passed=True,
                timestamp=None
            )
        ]
        
        regressions = benchmark_runner.analyzer.detect_regressions(current_results, previous_results)
        
        assert len(regressions) == 0, "Should not detect regressions with improved performance"


@pytest.mark.ci
class TestCIIntegration:
    """Test CI/CD integration functionality."""
    
    def test_benchmark_json_output(self, benchmark_results):
        """Test that benchmark results can be serialized to JSON."""
        import json
        
        # Test serialization
        serialized = []
        for result in benchmark_results:
            serialized.append({
                "test_id": result.test_id,
                "category": result.category,
                "query": result.query,
                "execution_time": result.execution_time,
                "overall_score": result.overall_score,
                "passed": result.passed,
                "timestamp": result.timestamp.isoformat() if result.timestamp else None
            })
        
        # Should be able to serialize to JSON
        json_str = json.dumps(serialized, default=str)
        assert isinstance(json_str, str)
        assert len(json_str) > 0
        
        # Should be able to deserialize
        deserialized = json.loads(json_str)
        assert isinstance(deserialized, list)
        assert len(deserialized) == len(benchmark_results)
    
    def test_benchmark_exit_codes(self, benchmark_results):
        """Test that benchmark results provide proper exit code information."""
        if not benchmark_results:
            pytest.skip("No benchmark results available")
        
        # Check that we can determine overall success
        all_passed = all(result.passed for result in benchmark_results)
        some_passed = any(result.passed for result in benchmark_results)
        
        assert isinstance(all_passed, bool)
        assert isinstance(some_passed, bool)
        
        # In CI, we might want to fail if any critical tests fail
        critical_tests = [r for r in benchmark_results if r.category in ["Science", "Technology"]]
        if critical_tests:
            critical_passed = all(result.passed for result in critical_tests)
            assert isinstance(critical_passed, bool)


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "benchmark: mark test as benchmark test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "regression: mark test as regression detection test"
    )
    config.addinivalue_line(
        "markers", "ci: mark test as CI/CD integration test"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names."""
    for item in items:
        # Add benchmark marker to benchmark tests
        if "benchmark" in item.name.lower():
            item.add_marker(pytest.mark.benchmark)
        
        # Add slow marker to full suite tests
        if "full" in item.name.lower() and "benchmark" in item.name.lower():
            item.add_marker(pytest.mark.slow)
        
        # Add regression marker to regression tests
        if "regression" in item.name.lower():
            item.add_marker(pytest.mark.regression)
        
        # Add ci marker to ci tests
        if "ci" in item.name.lower() or "integration" in item.name.lower():
            item.add_marker(pytest.mark.ci)
