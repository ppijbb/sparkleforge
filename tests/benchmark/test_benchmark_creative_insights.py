"""
Creative Insights Benchmark Tests

Tests for creative insights metrics including novelty score validation,
applicability assessment, cross-domain synthesis quality, and idea combination effectiveness.
"""

import pytest
import logging
from typing import Dict, List, Any
from pathlib import Path

from .benchmark_runner import BenchmarkRunner
from .benchmark_metrics import MetricsCollector, BenchmarkResult

logger = logging.getLogger(__name__)


class TestCreativeInsightsBenchmark:
    """Test creative insights benchmark functionality."""
    
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
    
    def test_creative_quality_measurement(self, metrics_collector):
        """Test creative insights quality measurement."""
        insights = [
            {
                "title": "AI-Powered Education Revolution",
                "type": "analogical",
                "novelty_score": 0.8,
                "applicability_score": 0.7,
                "confidence": 0.85,
                "description": "Applying AI concepts to revolutionize education"
            },
            {
                "title": "Cross-Domain Innovation",
                "type": "cross_domain",
                "novelty_score": 0.9,
                "applicability_score": 0.6,
                "confidence": 0.75,
                "description": "Combining healthcare and technology"
            }
        ]
        
        results = metrics_collector.measure_creative_quality(insights)
        
        assert len(results) == 2  # Should return novelty and applicability metrics
        
        # Check novelty metric
        novelty_result = next(r for r in results if r.name == "creative_novelty")
        assert novelty_result.value == 0.85  # Average of 0.8 and 0.9
        assert novelty_result.passed  # Should pass threshold
        assert novelty_result.category == "creative_insights"
        assert "insight_count" in novelty_result.metadata
        assert novelty_result.metadata["insight_count"] == 2
        
        # Check applicability metric
        applicability_result = next(r for r in results if r.name == "creative_applicability")
        assert applicability_result.value == 0.65  # Average of 0.7 and 0.6
        assert applicability_result.passed  # Should pass threshold
        assert applicability_result.category == "creative_insights"
    
    def test_creative_quality_with_no_insights(self, metrics_collector):
        """Test creative quality measurement with no insights."""
        results = metrics_collector.measure_creative_quality([])
        
        assert len(results) == 2  # Should still return both metrics
        
        for result in results:
            assert result.value == 0.0
            assert not result.passed
            assert "error" in result.metadata
    
    def test_creative_quality_with_low_scores(self, metrics_collector):
        """Test creative quality measurement with low scores."""
        insights = [
            {
                "title": "Low Novelty Idea",
                "type": "convergent",
                "novelty_score": 0.3,
                "applicability_score": 0.4,
                "confidence": 0.5
            },
            {
                "title": "Another Low Score Idea",
                "type": "divergent",
                "novelty_score": 0.2,
                "applicability_score": 0.3,
                "confidence": 0.4
            }
        ]
        
        results = metrics_collector.measure_creative_quality(insights)
        
        # Check novelty metric
        novelty_result = next(r for r in results if r.name == "creative_novelty")
        assert novelty_result.value == 0.25  # Average of 0.3 and 0.2
        assert not novelty_result.passed  # Should fail threshold
        
        # Check applicability metric
        applicability_result = next(r for r in results if r.name == "creative_applicability")
        assert applicability_result.value == 0.35  # Average of 0.4 and 0.3
        assert not applicability_result.passed  # Should fail threshold
    
    def test_creative_insights_benchmark_execution(self, benchmark_runner):
        """Test that creative insights benchmark can be executed."""
        results = benchmark_runner.run_creative_insights_benchmark()
        
        assert isinstance(results, list)
        assert len(results) > 0, "Should have at least one test result"
        
        # Check that all results are BenchmarkResult objects
        for result in results:
            assert isinstance(result, BenchmarkResult)
            assert result.test_id is not None
            assert result.category is not None
            assert result.query is not None
    
    def test_creative_insights_metrics_collection(self, benchmark_runner):
        """Test that creative insights metrics are properly collected."""
        test_case = {
            "id": "test-creative-001",
            "category": "Creative",
            "query": "Innovation in education",
            "expected": {
                "min_insights": 3,
                "min_novelty": 0.6
            }
        }
        
        extracted_metrics = {
            "creative_insights": [
                {
                    "title": "AI-Powered Learning",
                    "type": "analogical",
                    "novelty_score": 0.8,
                    "applicability_score": 0.7,
                    "confidence": 0.85
                },
                {
                    "title": "Cross-Domain Innovation",
                    "type": "cross_domain",
                    "novelty_score": 0.9,
                    "applicability_score": 0.6,
                    "confidence": 0.75
                }
            ],
            "sources": [],
            "execution_results": []
        }
        
        metrics = benchmark_runner._collect_metrics_for_test(test_case, extracted_metrics, 60.0)
        
        # Should have creative insights metrics
        metric_names = [m.name for m in metrics]
        assert "creative_novelty" in metric_names
        assert "creative_applicability" in metric_names
        
        # Check creative insights metrics
        novelty_metric = next(m for m in metrics if m.name == "creative_novelty")
        assert novelty_metric.category == "creative_insights"
        assert novelty_metric.value > 0
        
        applicability_metric = next(m for m in metrics if m.name == "creative_applicability")
        assert applicability_metric.category == "creative_insights"
        assert applicability_metric.value > 0
    
    def test_creative_insights_threshold_evaluation(self, benchmark_runner):
        """Test creative insights threshold evaluation."""
        test_case = {
            "id": "test-creative-threshold",
            "category": "Creative",
            "query": "Creative innovation",
            "expected": {
                "min_novelty": 0.7,
                "min_applicability": 0.6
            }
        }
        
        # Create metrics that should pass
        metrics = [
            type('Metric', (), {
                'name': 'creative_novelty',
                'value': 0.8,
                'threshold': 0.6,
                'passed': True,
                'category': 'creative_insights'
            })(),
            type('Metric', (), {
                'name': 'creative_applicability',
                'value': 0.7,
                'threshold': 0.6,
                'passed': True,
                'category': 'creative_insights'
            })()
        ]
        
        passed = benchmark_runner._evaluate_test_pass(metrics, test_case['expected'])
        assert passed, "Test should pass with good creative metrics"
        
        # Create metrics that should fail
        failing_metrics = [
            type('Metric', (), {
                'name': 'creative_novelty',
                'value': 0.5,
                'threshold': 0.6,
                'passed': False,
                'category': 'creative_insights'
            })(),
            type('Metric', (), {
                'name': 'creative_applicability',
                'value': 0.4,
                'threshold': 0.6,
                'passed': False,
                'category': 'creative_insights'
            })()
        ]
        
        failed = benchmark_runner._evaluate_test_pass(failing_metrics, test_case['expected'])
        assert not failed, "Test should fail with poor creative metrics"
    
    def test_creative_insights_metadata(self, metrics_collector):
        """Test that creative insights metrics include proper metadata."""
        insights = [
            {
                "title": "Test Insight",
                "type": "analogical",
                "novelty_score": 0.8,
                "applicability_score": 0.7,
                "confidence": 0.85
            }
        ]
        
        results = metrics_collector.measure_creative_quality(insights)
        
        for result in results:
            assert "insight_count" in result.metadata
            assert "insight_details" in result.metadata
            assert result.metadata["insight_count"] == 1
            assert len(result.metadata["insight_details"]) == 1
            
            insight_detail = result.metadata["insight_details"][0]
            assert insight_detail["title"] == "Test Insight"
            assert insight_detail["type"] == "analogical"
    
    def test_creative_insights_benchmark_summary(self, benchmark_runner):
        """Test creative insights benchmark summary generation."""
        mock_results = [
            BenchmarkResult(
                test_id="creative-001",
                category="Creative",
                query="Innovation ideas",
                execution_time=120.0,
                metrics=[],
                overall_score=0.8,
                passed=True,
                timestamp=None
            ),
            BenchmarkResult(
                test_id="creative-002",
                category="Health",
                query="Health innovation",
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
        
        # Check category breakdown
        breakdown = summary["category_breakdown"]
        assert "Creative" in breakdown
        assert "Health" in breakdown
    
    def test_creative_insights_different_types(self, metrics_collector):
        """Test creative insights with different insight types."""
        insights = [
            {
                "title": "Analogical Insight",
                "type": "analogical",
                "novelty_score": 0.8,
                "applicability_score": 0.7
            },
            {
                "title": "Cross-Domain Insight",
                "type": "cross_domain",
                "novelty_score": 0.9,
                "applicability_score": 0.6
            },
            {
                "title": "Lateral Insight",
                "type": "lateral",
                "novelty_score": 0.7,
                "applicability_score": 0.8
            },
            {
                "title": "Convergent Insight",
                "type": "convergent",
                "novelty_score": 0.6,
                "applicability_score": 0.9
            },
            {
                "title": "Divergent Insight",
                "type": "divergent",
                "novelty_score": 0.95,
                "applicability_score": 0.5
            }
        ]
        
        results = metrics_collector.measure_creative_quality(insights)
        
        # Should handle all insight types
        assert len(results) == 2  # Novelty and applicability
        
        # Check that all insights are included in metadata
        for result in results:
            assert result.metadata["insight_count"] == 5
            assert len(result.metadata["insight_details"]) == 5
            
            # Check that all types are represented
            types = [detail["type"] for detail in result.metadata["insight_details"]]
            expected_types = ["analogical", "cross_domain", "lateral", "convergent", "divergent"]
            for expected_type in expected_types:
                assert expected_type in types
    
    def test_creative_insights_confidence_scoring(self, metrics_collector):
        """Test creative insights confidence scoring."""
        insights = [
            {
                "title": "High Confidence Insight",
                "type": "analogical",
                "novelty_score": 0.8,
                "applicability_score": 0.7,
                "confidence": 0.9
            },
            {
                "title": "Low Confidence Insight",
                "type": "cross_domain",
                "novelty_score": 0.6,
                "applicability_score": 0.5,
                "confidence": 0.3
            }
        ]
        
        results = metrics_collector.measure_creative_quality(insights)
        
        # Check that confidence scores are included in metadata
        for result in results:
            insight_details = result.metadata["insight_details"]
            assert len(insight_details) == 2
            
            # Find insights by title
            high_conf_insight = next(d for d in insight_details if d["title"] == "High Confidence Insight")
            low_conf_insight = next(d for d in insight_details if d["title"] == "Low Confidence Insight")
            
            assert high_conf_insight["confidence"] == 0.9
            assert low_conf_insight["confidence"] == 0.3
    
    @pytest.mark.slow
    def test_full_creative_insights_benchmark(self, benchmark_runner):
        """Test full creative insights benchmark execution (slow test)."""
        # This test actually runs the CLI and may take time
        results = benchmark_runner.run_creative_insights_benchmark()
        
        assert len(results) > 0, "Should have results from creative insights benchmark"
        
        # Check that we have results from creative-focused categories
        categories = {result.category for result in results}
        expected_categories = {"Creative", "Health"}
        assert categories.intersection(expected_categories), "Should have results from creative categories"
    
    def test_creative_insights_benchmark_configuration(self, benchmark_runner):
        """Test that creative insights benchmark uses correct configuration."""
        # Check that the benchmark runner has the right configuration
        assert benchmark_runner.config is not None
        assert "test_cases" in benchmark_runner.config
        
        # Check that creative test cases exist
        test_cases = benchmark_runner.config["test_cases"]
        creative_cases = [tc for tc in test_cases if tc.get("category") == "Creative"]
        assert len(creative_cases) > 0, "Should have creative test cases in configuration"
        
        # Check that creative cases have expected metrics
        for case in creative_cases:
            expected = case.get("expected", {})
            assert "min_insights" in expected, "Creative cases should have min_insights expectation"
