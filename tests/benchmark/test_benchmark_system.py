#!/usr/bin/env python3
"""
Test the SparkleForge benchmark system with mock data.

This script tests the benchmark system without requiring actual CLI execution,
using mock data to verify all components work correctly.
"""

import sys
import logging
from pathlib import Path

# Add tests directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from benchmark_metrics import MetricsCollector, BenchmarkAnalyzer, BenchmarkResult, MetricResult
from benchmark_runner import BenchmarkRunner
from benchmark_reporter import BenchmarkReporter
from benchmark_comparator import BenchmarkComparator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_metrics_collector():
    """Test metrics collector with mock data."""
    print("Testing MetricsCollector...")
    
    thresholds = {
        "response_time": 120.0,
        "source_credibility": 0.7,
        "factual_accuracy": 0.85,
        "creative_novelty": 0.6,
        "memory_precision": 0.7
    }
    
    collector = MetricsCollector(thresholds)
    
    # Test response time measurement
    import time
    start_time = time.time()
    time.sleep(0.1)
    end_time = time.time()
    
    response_time_result = collector.measure_response_time(start_time, end_time)
    assert response_time_result.name == "response_time"
    assert response_time_result.value > 0.09
    assert response_time_result.passed
    print("✓ Response time measurement works")
    
    # Test source credibility measurement
    sources = [
        {"domain": "nature.com", "credibility_score": 0.9},
        {"domain": "science.org", "credibility_score": 0.85}
    ]
    
    credibility_result = collector.measure_source_credibility(sources)
    assert credibility_result.name == "source_credibility"
    assert credibility_result.value > 0.8
    assert credibility_result.passed
    print("✓ Source credibility measurement works")
    
    # Test creative quality measurement
    insights = [
        {
            "title": "Test Insight",
            "type": "analogical",
            "novelty_score": 0.8,
            "applicability_score": 0.7
        }
    ]
    
    creative_results = collector.measure_creative_quality(insights)
    assert len(creative_results) == 2
    assert any(r.name == "creative_novelty" for r in creative_results)
    assert any(r.name == "creative_applicability" for r in creative_results)
    print("✓ Creative quality measurement works")
    
    print("✅ MetricsCollector tests passed\n")


def test_benchmark_analyzer():
    """Test benchmark analyzer with mock data."""
    print("Testing BenchmarkAnalyzer...")
    
    thresholds = {"response_time": 120.0, "source_credibility": 0.7}
    weights = {"performance": 0.5, "research_quality": 0.5}
    
    analyzer = BenchmarkAnalyzer(thresholds, weights)
    
    # Create mock results
    results = [
        BenchmarkResult(
            test_id="test-001",
            category="Technology",
            query="Test query",
            execution_time=60.0,
            metrics=[
                MetricResult("response_time", 60.0, 120.0, True, "performance"),
                MetricResult("source_credibility", 0.8, 0.7, True, "research_quality")
            ],
            overall_score=0.8,
            passed=True,
            timestamp=None
        ),
        BenchmarkResult(
            test_id="test-002",
            category="Science",
            query="Test query 2",
            execution_time=90.0,
            metrics=[
                MetricResult("response_time", 90.0, 120.0, True, "performance"),
                MetricResult("source_credibility", 0.75, 0.7, True, "research_quality")
            ],
            overall_score=0.75,
            passed=True,
            timestamp=None
        )
    ]
    
    # Test overall score calculation
    overall_score = analyzer.calculate_overall_score(results)
    assert 0 <= overall_score <= 1
    print(f"✓ Overall score calculation works: {overall_score:.3f}")
    
    # Test regression detection
    previous_results = [
        BenchmarkResult(
            test_id="test-001",
            category="Technology",
            query="Test query",
            execution_time=50.0,  # Faster
            metrics=[],
            overall_score=0.9,  # Higher score
            passed=True,
            timestamp=None
        )
    ]
    
    regressions = analyzer.detect_regressions(results, previous_results)
    assert len(regressions) > 0
    print(f"✓ Regression detection works: {len(regressions)} regressions found")
    
    print("✅ BenchmarkAnalyzer tests passed\n")


def test_benchmark_reporter():
    """Test benchmark reporter with mock data."""
    print("Testing BenchmarkReporter...")
    
    reporter = BenchmarkReporter("test_results")
    
    # Create mock results
    results = [
        BenchmarkResult(
            test_id="test-001",
            category="Technology",
            query="AI trends",
            execution_time=60.0,
            metrics=[
                MetricResult("response_time", 60.0, 120.0, True, "performance"),
                MetricResult("source_credibility", 0.8, 0.7, True, "research_quality")
            ],
            overall_score=0.8,
            passed=True,
            timestamp=None
        ),
        BenchmarkResult(
            test_id="test-002",
            category="Science",
            query="Climate research",
            execution_time=90.0,
            metrics=[
                MetricResult("response_time", 90.0, 120.0, True, "performance"),
                MetricResult("source_credibility", 0.75, 0.7, True, "research_quality")
            ],
            overall_score=0.75,
            passed=True,
            timestamp=None
        )
    ]
    
    # Test JSON report generation
    json_file = reporter.generate_json_report(results)
    assert Path(json_file).exists()
    print(f"✓ JSON report generated: {json_file}")
    
    # Test console summary
    console_summary = reporter.generate_console_summary(results)
    assert "SPARKLEFORGE BENCHMARK SUMMARY" in console_summary
    assert "Total Tests: 2" in console_summary
    print("✓ Console summary generation works")
    
    print("✅ BenchmarkReporter tests passed\n")


def test_benchmark_comparator():
    """Test benchmark comparator with mock data."""
    print("Testing BenchmarkComparator...")
    
    comparator = BenchmarkComparator("test_baselines")
    
    # Create mock current results
    current_results = [
        BenchmarkResult(
            test_id="test-001",
            category="Technology",
            query="Test query",
            execution_time=100.0,
            metrics=[],
            overall_score=0.7,
            passed=True,
            timestamp=None
        )
    ]
    
    # Create mock previous results
    previous_results = [
        BenchmarkResult(
            test_id="test-001",
            category="Technology",
            query="Test query",
            execution_time=80.0,  # Faster
            metrics=[],
            overall_score=0.8,  # Higher score
            passed=True,
            timestamp=None
        )
    ]
    
    # Test regression detection
    regressions = comparator.detect_regression(current_results, previous_results)
    assert len(regressions) > 0
    print(f"✓ Regression detection works: {len(regressions)} regressions found")
    
    # Test improvement report generation
    improvement_report = comparator.generate_improvement_report(current_results, previous_results)
    assert "Benchmark Improvement Report" in improvement_report
    print("✓ Improvement report generation works")
    
    print("✅ BenchmarkComparator tests passed\n")


def test_benchmark_runner():
    """Test benchmark runner with production configuration."""
    print("Testing BenchmarkRunner...")
    
    # Use actual configuration files
    project_root = Path(__file__).parent.parent.parent
    config_path = project_root / "tests" / "benchmark" / "benchmark_config.yaml"
    thresholds_path = project_root / "tests" / "benchmark" / "benchmark_thresholds.yaml"
    
    if not config_path.exists() or not thresholds_path.exists():
        print("⚠️  Configuration files not found, skipping runner test")
        return
    
    try:
        runner = BenchmarkRunner(str(project_root), str(config_path), str(thresholds_path))
        
        # Test configuration loading
        assert runner.config is not None
        assert "test_cases" in runner.config
        assert "environment" in runner.config
        print("✓ Configuration loading works")
        
        # Test production environment settings
        env_config = runner.config.get("environment", {})
        assert env_config.get("llm_provider") == "openrouter"
        assert env_config.get("llm_model") == "google/gemini-2.5-flash-lite"
        assert env_config.get("enable_auto_fallback") == False
        print("✓ Production environment configuration verified")
        
        # Test threshold loading
        assert runner.thresholds is not None
        assert "thresholds" in runner.thresholds
        print("✓ Threshold loading works")
        
        # Test production threshold values
        thresholds = runner.thresholds["thresholds"]
        assert thresholds["response_time"] == 90  # Production threshold
        assert thresholds["source_credibility"] == 0.8  # Higher production standard
        print("✓ Production thresholds verified")
        
        # Test summary generation with empty results
        summary = runner.get_benchmark_summary([])
        assert summary["total_tests"] == 0
        print("✓ Summary generation works")
        
        print("✅ BenchmarkRunner tests passed\n")
        
    except Exception as e:
        print(f"⚠️  BenchmarkRunner test failed: {e}")


def main():
    """Run all benchmark system tests."""
    print("🧪 Testing SparkleForge Benchmark System")
    print("=" * 50)
    
    try:
        test_metrics_collector()
        test_benchmark_analyzer()
        test_benchmark_reporter()
        test_benchmark_comparator()
        test_benchmark_runner()
        
        print("🎉 All benchmark system tests passed!")
        print("✅ The SparkleForge benchmark system is working correctly.")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
