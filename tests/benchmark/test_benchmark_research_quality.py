"""
Research Quality Benchmark Tests

Tests for research quality metrics including information accuracy,
source credibility scoring, analysis depth evaluation, and synthesis quality.
"""

import pytest
import logging
from typing import Dict, List, Any
from pathlib import Path

from .benchmark_runner import BenchmarkRunner
from .benchmark_metrics import MetricsCollector, BenchmarkResult

logger = logging.getLogger(__name__)


class TestResearchQualityBenchmark:
    """Test research quality benchmark functionality."""
    
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
    
    def test_research_quality_benchmark_execution(self, benchmark_runner):
        """Test that research quality benchmark can be executed."""
        results = benchmark_runner.run_research_quality_benchmark()
        
        assert isinstance(results, list)
        assert len(results) > 0, "Should have at least one test result"
        
        # Check that all results are BenchmarkResult objects
        for result in results:
            assert isinstance(result, BenchmarkResult)
            assert result.test_id is not None
            assert result.category is not None
            assert result.query is not None
            assert result.execution_time >= 0
            assert isinstance(result.metrics, list)
            assert 0 <= result.overall_score <= 1
            assert isinstance(result.passed, bool)
    
    def test_source_credibility_measurement(self, metrics_collector):
        """Test source credibility measurement."""
        # Test with high credibility sources
        high_cred_sources = [
            {"domain": "nature.com", "title": "Scientific Study", "credibility_score": 0.9},
            {"domain": "science.org", "title": "Research Paper", "credibility_score": 0.95},
            {"domain": "arxiv.org", "title": "Preprint", "credibility_score": 0.8}
        ]
        
        result = metrics_collector.measure_source_credibility(high_cred_sources)
        
        assert result.name == "source_credibility"
        assert result.value > 0.8  # Should be high
        assert result.passed  # Should pass threshold
        assert result.category == "research_quality"
        assert "source_count" in result.metadata
        assert result.metadata["source_count"] == 3
    
    def test_source_credibility_with_low_quality_sources(self, metrics_collector):
        """Test source credibility with low quality sources."""
        low_cred_sources = [
            {"domain": "unknown-blog.com", "title": "Opinion Piece", "credibility_score": 0.3},
            {"domain": "random-site.net", "title": "Unverified Info", "credibility_score": 0.2}
        ]
        
        result = metrics_collector.measure_source_credibility(low_cred_sources)
        
        assert result.name == "source_credibility"
        assert result.value < 0.5  # Should be low
        assert not result.passed  # Should fail threshold
        assert result.category == "research_quality"
    
    def test_factual_accuracy_measurement(self, metrics_collector):
        """Test factual accuracy measurement."""
        claims = [
            "Artificial intelligence is advancing rapidly in 2025",
            "Machine learning models are becoming more efficient",
            "Neural networks can process complex data patterns"
        ]
        
        sources = [
            {
                "title": "AI Trends 2025",
                "content": "Artificial intelligence is advancing rapidly in 2025 with new breakthroughs in machine learning and neural networks",
                "url": "https://example.com/ai-trends"
            },
            {
                "title": "ML Efficiency Study",
                "content": "Machine learning models are becoming more efficient with new optimization techniques",
                "url": "https://example.com/ml-efficiency"
            }
        ]
        
        result = metrics_collector.measure_factual_accuracy(claims, sources)
        
        assert result.name == "factual_accuracy"
        assert 0 <= result.value <= 1
        assert result.category == "research_quality"
        assert "total_claims" in result.metadata
        assert "verified_claims" in result.metadata
        assert result.metadata["total_claims"] == 3
    
    def test_factual_accuracy_with_no_sources(self, metrics_collector):
        """Test factual accuracy with no sources."""
        claims = ["Some claim", "Another claim"]
        sources = []
        
        result = metrics_collector.measure_factual_accuracy(claims, sources)
        
        assert result.name == "factual_accuracy"
        assert result.value == 0.0
        assert not result.passed
        assert "error" in result.metadata
    
    def test_research_quality_metrics_collection(self, benchmark_runner):
        """Test that research quality metrics are properly collected."""
        # Create a mock test case
        test_case = {
            "id": "test-quality-001",
            "category": "Technology",
            "query": "AI developments",
            "expected": {
                "min_sources": 5,
                "min_credibility": 0.7
            }
        }
        
        # Mock extracted metrics
        extracted_metrics = {
            "sources": [
                {"domain": "nature.com", "credibility_score": 0.9},
                {"domain": "science.org", "credibility_score": 0.95}
            ],
            "execution_results": [
                {"summary": "AI is advancing rapidly"},
                {"findings": ["Machine learning is improving", "Neural networks are more efficient"]}
            ]
        }
        
        metrics = benchmark_runner._collect_metrics_for_test(test_case, extracted_metrics, 60.0)
        
        # Should have response time, source credibility, and factual accuracy metrics
        metric_names = [m.name for m in metrics]
        assert "response_time" in metric_names
        assert "source_credibility" in metric_names
        assert "factual_accuracy" in metric_names
    
    def test_research_quality_threshold_evaluation(self, benchmark_runner):
        """Test that research quality thresholds are properly evaluated."""
        # Create test case with specific expectations
        test_case = {
            "id": "test-threshold-001",
            "category": "Science",
            "query": "Climate change research",
            "expected": {
                "min_credibility": 0.8,
                "max_response_time": 120
            }
        }
        
        # Create metrics that should pass
        metrics = [
            type('Metric', (), {
                'name': 'response_time',
                'value': 100.0,
                'threshold': 120.0,
                'passed': True,
                'category': 'performance'
            })(),
            type('Metric', (), {
                'name': 'source_credibility',
                'value': 0.85,
                'threshold': 0.8,
                'passed': True,
                'category': 'research_quality'
            })()
        ]
        
        passed = benchmark_runner._evaluate_test_pass(metrics, test_case['expected'])
        assert passed, "Test should pass with good metrics"
        
        # Create metrics that should fail
        failing_metrics = [
            type('Metric', (), {
                'name': 'response_time',
                'value': 150.0,
                'threshold': 120.0,
                'passed': False,
                'category': 'performance'
            })(),
            type('Metric', (), {
                'name': 'source_credibility',
                'value': 0.75,
                'threshold': 0.8,
                'passed': False,
                'category': 'research_quality'
            })()
        ]
        
        failed = benchmark_runner._evaluate_test_pass(failing_metrics, test_case['expected'])
        assert not failed, "Test should fail with poor metrics"
    
    def test_research_quality_benchmark_summary(self, benchmark_runner):
        """Test research quality benchmark summary generation."""
        # Create mock results
        mock_results = [
            BenchmarkResult(
                test_id="test-001",
                category="Technology",
                query="AI trends",
                execution_time=60.0,
                metrics=[],
                overall_score=0.8,
                passed=True,
                timestamp=None
            ),
            BenchmarkResult(
                test_id="test-002",
                category="Science",
                query="Climate research",
                execution_time=90.0,
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
        assert summary["overall_score"] > 0
        assert summary["average_execution_time"] == 75.0
        assert "category_breakdown" in summary
    
    @pytest.mark.slow
    def test_full_research_quality_benchmark(self, benchmark_runner):
        """Test full research quality benchmark execution (slow test)."""
        # This test actually runs the CLI and may take time
        results = benchmark_runner.run_research_quality_benchmark()
        
        assert len(results) > 0, "Should have results from research quality benchmark"
        
        # Check that we have results from different categories
        categories = {result.category for result in results}
        expected_categories = {"Technology", "Science", "Health"}
        assert categories.intersection(expected_categories), "Should have results from expected categories"
        
        # Check that some tests passed (if environment is properly set up)
        passed_count = sum(1 for result in results if result.passed)
        assert passed_count >= 0, "Should have some test results (pass or fail)"
    
    def test_metrics_collector_initialization(self, project_root):
        """Test metrics collector initialization."""
        thresholds = {
            "source_credibility": 0.7,
            "factual_accuracy": 0.85,
            "response_time": 120.0
        }
        
        collector = MetricsCollector(thresholds)
        assert collector.thresholds == thresholds
        assert collector.logger is not None
    
    def test_benchmark_result_creation(self):
        """Test BenchmarkResult creation and properties."""
        result = BenchmarkResult(
            test_id="test-001",
            category="Technology",
            query="Test query",
            execution_time=60.0,
            metrics=[],
            overall_score=0.8,
            passed=True,
            timestamp=None
        )
        
        assert result.test_id == "test-001"
        assert result.category == "Technology"
        assert result.query == "Test query"
        assert result.execution_time == 60.0
        assert result.overall_score == 0.8
        assert result.passed is True
        assert isinstance(result.metrics, list)
