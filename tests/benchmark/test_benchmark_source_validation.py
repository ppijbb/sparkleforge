"""
Source Validation Benchmark Tests

Tests for source validation metrics including credibility scoring accuracy,
fact-checking effectiveness, cross-verification capability, and citation completeness.
"""

import pytest
import logging
from typing import Dict, List, Any
from pathlib import Path

from .benchmark_runner import BenchmarkRunner
from .benchmark_metrics import MetricsCollector, BenchmarkResult

logger = logging.getLogger(__name__)


class TestSourceValidationBenchmark:
    """Test source validation benchmark functionality."""
    
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
    
    def test_source_credibility_scoring(self, metrics_collector):
        """Test source credibility scoring with various domain types."""
        # Test high credibility sources
        high_cred_sources = [
            {"domain": "nature.com", "title": "Scientific Study", "credibility_score": 0.95},
            {"domain": "science.org", "title": "Research Paper", "credibility_score": 0.9},
            {"domain": "arxiv.org", "title": "Preprint", "credibility_score": 0.85}
        ]
        
        result = metrics_collector.measure_source_credibility(high_cred_sources)
        
        assert result.name == "source_credibility"
        assert result.value > 0.8
        assert result.passed
        assert result.category == "research_quality"
        assert result.metadata["source_count"] == 3
        assert result.metadata["min_credibility"] > 0.8
        assert result.metadata["max_credibility"] > 0.9
    
    def test_domain_credibility_estimation(self, metrics_collector):
        """Test domain-based credibility estimation."""
        # Test with domains that should have different credibility scores
        test_domains = [
            "nature.com",  # High credibility
            "science.org",  # High credibility
            "wikipedia.org",  # Medium credibility
            "unknown-blog.com",  # Low credibility
            "random-site.net"  # Low credibility
        ]
        
        for domain in test_domains:
            credibility = metrics_collector._estimate_domain_credibility(domain)
            assert 0 <= credibility <= 1, f"Credibility should be between 0 and 1 for {domain}"
            
            if domain in ["nature.com", "science.org"]:
                assert credibility > 0.8, f"High credibility domain {domain} should score > 0.8"
            elif domain == "wikipedia.org":
                assert 0.6 <= credibility <= 0.8, f"Medium credibility domain {domain} should score 0.6-0.8"
            else:
                assert credibility <= 0.6, f"Low credibility domain {domain} should score <= 0.6"
    
    def test_factual_accuracy_verification(self, metrics_collector):
        """Test factual accuracy verification against sources."""
        claims = [
            "Climate change is caused by greenhouse gas emissions",
            "Renewable energy sources are becoming more cost-effective",
            "Carbon dioxide levels have increased significantly"
        ]
        
        sources = [
            {
                "title": "Climate Science Report",
                "content": "Climate change is primarily caused by greenhouse gas emissions from human activities. Carbon dioxide levels have increased significantly over the past century.",
                "url": "https://example.com/climate-report"
            },
            {
                "title": "Renewable Energy Study",
                "content": "Renewable energy sources like solar and wind are becoming more cost-effective and competitive with fossil fuels.",
                "url": "https://example.com/renewable-study"
            }
        ]
        
        result = metrics_collector.measure_factual_accuracy(claims, sources)
        
        assert result.name == "factual_accuracy"
        assert result.value > 0.5  # Should have some accuracy
        assert result.category == "research_quality"
        assert result.metadata["total_claims"] == 3
        assert result.metadata["verified_claims"] > 0
        
        # Check verification details
        verification_details = result.metadata["verification_details"]
        assert len(verification_details) == 3
        
        for detail in verification_details:
            assert "claim" in detail
            assert "verified" in detail
            assert "supporting_sources" in detail
    
    def test_cross_verification_capability(self, metrics_collector):
        """Test cross-verification capability with multiple sources."""
        claims = ["AI is advancing rapidly in 2025"]
        
        # Multiple sources with similar information
        sources = [
            {
                "title": "AI Trends Report 2025",
                "content": "Artificial intelligence is advancing rapidly in 2025 with new breakthroughs",
                "url": "https://example.com/ai-trends"
            },
            {
                "title": "Technology Review",
                "content": "AI technology is advancing rapidly this year with significant progress",
                "url": "https://example.com/tech-review"
            },
            {
                "title": "Research Journal",
                "content": "Artificial intelligence shows rapid advancement in 2025",
                "url": "https://example.com/research-journal"
            }
        ]
        
        result = metrics_collector.measure_factual_accuracy(claims, sources)
        
        # Should have high accuracy with multiple supporting sources
        assert result.value > 0.8
        assert result.passed
        
        # Check that multiple sources are found
        verification_details = result.metadata["verification_details"]
        claim_detail = verification_details[0]
        assert len(claim_detail["supporting_sources"]) > 1
    
    def test_citation_completeness(self, metrics_collector):
        """Test citation completeness measurement."""
        # This would typically be measured by checking if sources have proper citations
        # For now, we'll test the source credibility which is related
        sources_with_citations = [
            {
                "domain": "nature.com",
                "title": "Study with Citations",
                "credibility_score": 0.9,
                "citations": ["ref1", "ref2", "ref3"]
            },
            {
                "domain": "science.org",
                "title": "Another Study",
                "credibility_score": 0.85,
                "citations": ["ref4", "ref5"]
            }
        ]
        
        result = metrics_collector.measure_source_credibility(sources_with_citations)
        
        # High credibility sources typically have good citations
        assert result.value > 0.8
        assert result.passed
    
    def test_source_validation_benchmark_execution(self, benchmark_runner):
        """Test that source validation benchmark can be executed."""
        results = benchmark_runner.run_source_validation_benchmark()
        
        assert isinstance(results, list)
        assert len(results) > 0, "Should have at least one test result"
        
        # Check that all results are from science/health categories
        for result in results:
            assert isinstance(result, BenchmarkResult)
            assert result.category in ["Science", "Health"]
    
    def test_source_validation_metrics_collection(self, benchmark_runner):
        """Test that source validation metrics are properly collected."""
        test_case = {
            "id": "test-validation-001",
            "category": "Science",
            "query": "Climate change research",
            "expected": {
                "min_credibility": 0.8,
                "min_sources": 10
            }
        }
        
        extracted_metrics = {
            "sources": [
                {"domain": "nature.com", "credibility_score": 0.9},
                {"domain": "science.org", "credibility_score": 0.85},
                {"domain": "arxiv.org", "credibility_score": 0.8}
            ],
            "execution_results": [
                {"summary": "Climate change is a serious issue"},
                {"findings": ["Temperature is rising", "Sea levels are increasing"]}
            ]
        }
        
        metrics = benchmark_runner._collect_metrics_for_test(test_case, extracted_metrics, 60.0)
        
        # Should have source credibility and factual accuracy metrics
        metric_names = [m.name for m in metrics]
        assert "source_credibility" in metric_names
        assert "factual_accuracy" in metric_names
        
        # Check source credibility metric
        credibility_metric = next(m for m in metrics if m.name == "source_credibility")
        assert credibility_metric.category == "research_quality"
        assert credibility_metric.value > 0.8  # Should be high with good sources
    
    def test_source_validation_threshold_evaluation(self, benchmark_runner):
        """Test source validation threshold evaluation."""
        test_case = {
            "id": "test-validation-threshold",
            "category": "Science",
            "query": "Scientific research",
            "expected": {
                "min_credibility": 0.8
            }
        }
        
        # Create metrics that should pass
        metrics = [
            type('Metric', (), {
                'name': 'source_credibility',
                'value': 0.85,
                'threshold': 0.8,
                'passed': True,
                'category': 'research_quality'
            })(),
            type('Metric', (), {
                'name': 'factual_accuracy',
                'value': 0.9,
                'threshold': 0.85,
                'passed': True,
                'category': 'research_quality'
            })()
        ]
        
        passed = benchmark_runner._evaluate_test_pass(metrics, test_case['expected'])
        assert passed, "Test should pass with good validation metrics"
        
        # Create metrics that should fail
        failing_metrics = [
            type('Metric', (), {
                'name': 'source_credibility',
                'value': 0.6,
                'threshold': 0.8,
                'passed': False,
                'category': 'research_quality'
            })(),
            type('Metric', (), {
                'name': 'factual_accuracy',
                'value': 0.7,
                'threshold': 0.85,
                'passed': False,
                'category': 'research_quality'
            })()
        ]
        
        failed = benchmark_runner._evaluate_test_pass(failing_metrics, test_case['expected'])
        assert not failed, "Test should fail with poor validation metrics"
    
    def test_keyword_extraction(self, metrics_collector):
        """Test keyword extraction from text."""
        text = "Artificial intelligence is advancing rapidly in 2025 with new machine learning techniques"
        keywords = metrics_collector._extract_keywords(text)
        
        assert isinstance(keywords, list)
        assert len(keywords) > 0
        assert "artificial" in keywords
        assert "intelligence" in keywords
        assert "advancing" in keywords
        assert "machine" in keywords
        assert "learning" in keywords
        
        # Should filter out common stop words
        assert "is" not in keywords
        assert "in" not in keywords
        assert "with" not in keywords
        assert "new" not in keywords
    
    def test_text_similarity_calculation(self, metrics_collector):
        """Test text similarity calculation."""
        text1 = "Artificial intelligence is advancing rapidly"
        text2 = "AI technology is advancing quickly"
        text3 = "Climate change is a serious problem"
        
        similarity_high = metrics_collector._calculate_similarity(text1, text2)
        similarity_low = metrics_collector._calculate_similarity(text1, text3)
        
        assert 0 <= similarity_high <= 1
        assert 0 <= similarity_low <= 1
        assert similarity_high > similarity_low  # Similar texts should have higher similarity
    
    def test_source_validation_benchmark_summary(self, benchmark_runner):
        """Test source validation benchmark summary generation."""
        mock_results = [
            BenchmarkResult(
                test_id="validation-001",
                category="Science",
                query="Climate research",
                execution_time=120.0,
                metrics=[],
                overall_score=0.85,
                passed=True,
                timestamp=None
            ),
            BenchmarkResult(
                test_id="validation-002",
                category="Health",
                query="Medical research",
                execution_time=150.0,
                metrics=[],
                overall_score=0.7,
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
        assert "Science" in breakdown
        assert "Health" in breakdown
    
    def test_empty_sources_handling(self, metrics_collector):
        """Test handling of empty sources list."""
        result = metrics_collector.measure_source_credibility([])
        
        assert result.value == 0.0
        assert not result.passed
        assert "error" in result.metadata
        assert result.metadata["error"] == "No sources found"
    
    def test_empty_claims_handling(self, metrics_collector):
        """Test handling of empty claims list."""
        result = metrics_collector.measure_factual_accuracy([], [])
        
        assert result.value == 0.0
        assert not result.passed
        assert "error" in result.metadata
        assert "No claims or sources to verify" in result.metadata["error"]
    
    @pytest.mark.slow
    def test_full_source_validation_benchmark(self, benchmark_runner):
        """Test full source validation benchmark execution (slow test)."""
        # This test actually runs the CLI and may take time
        results = benchmark_runner.run_source_validation_benchmark()
        
        assert len(results) > 0, "Should have results from source validation benchmark"
        
        # Check that we have results from validation-focused categories
        categories = {result.category for result in results}
        expected_categories = {"Science", "Health"}
        assert categories.intersection(expected_categories), "Should have results from validation categories"
    
    def test_source_validation_benchmark_configuration(self, benchmark_runner):
        """Test that source validation benchmark uses correct configuration."""
        # Check that the benchmark runner has the right configuration
        assert benchmark_runner.config is not None
        assert "test_cases" in benchmark_runner.config
        
        # Check that science/health test cases exist
        test_cases = benchmark_runner.config["test_cases"]
        validation_cases = [tc for tc in test_cases if tc.get("category") in ["Science", "Health"]]
        assert len(validation_cases) > 0, "Should have validation test cases in configuration"
        
        # Check that validation cases have expected metrics
        for case in validation_cases:
            expected = case.get("expected", {})
            assert "min_credibility" in expected, "Validation cases should have min_credibility expectation"
