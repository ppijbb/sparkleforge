"""
Memory and Learning Benchmark Tests

Tests for memory and learning metrics including similar research retrieval accuracy,
user preference learning, recommendation quality, and pattern recognition capability.
"""

import pytest
import logging
from typing import Dict, List, Any
from pathlib import Path

from .benchmark_runner import BenchmarkRunner
from .benchmark_metrics import MetricsCollector, BenchmarkResult

logger = logging.getLogger(__name__)


class TestMemoryLearningBenchmark:
    """Test memory and learning benchmark functionality."""
    
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
    
    def test_memory_accuracy_measurement(self, metrics_collector):
        """Test memory retrieval accuracy measurement."""
        # Simulate retrieved memories
        retrieved = [
            {
                "title": "AI Trends in 2025",
                "content": "Artificial intelligence is advancing rapidly with new machine learning techniques",
                "similarity_score": 0.85
            },
            {
                "title": "Machine Learning Advances",
                "content": "Deep learning models are becoming more efficient and accurate",
                "similarity_score": 0.78
            }
        ]
        
        # Simulate expected/ground truth memories
        expected = [
            {
                "title": "AI Development Trends",
                "content": "Artificial intelligence development is progressing quickly with advanced machine learning",
                "similarity_score": 0.9
            },
            {
                "title": "Deep Learning Progress",
                "content": "Deep learning algorithms are improving in efficiency and accuracy",
                "similarity_score": 0.88
            }
        ]
        
        result = metrics_collector.measure_memory_accuracy(retrieved, expected)
        
        assert result.name == "memory_precision"
        assert 0 <= result.value <= 1
        assert result.category == "memory_learning"
        assert result.metadata["total_retrieved"] == 2
        assert result.metadata["correct_retrievals"] >= 0
        assert len(result.metadata["retrieval_details"]) == 2
    
    def test_memory_accuracy_with_no_retrieved(self, metrics_collector):
        """Test memory accuracy with no retrieved memories."""
        result = metrics_collector.measure_memory_accuracy([], [])
        
        assert result.value == 0.0
        assert not result.passed
        assert "error" in result.metadata
        assert result.metadata["error"] == "No retrieved memories"
    
    def test_memory_accuracy_with_perfect_match(self, metrics_collector):
        """Test memory accuracy with perfect matches."""
        retrieved = [
            {
                "title": "AI Trends",
                "content": "Artificial intelligence is advancing rapidly"
            }
        ]
        
        expected = [
            {
                "title": "AI Trends",
                "content": "Artificial intelligence is advancing rapidly"
            }
        ]
        
        result = metrics_collector.measure_memory_accuracy(retrieved, expected)
        
        # Should have high accuracy with perfect match
        assert result.value > 0.8
        assert result.passed
        
        # Check retrieval details
        details = result.metadata["retrieval_details"]
        assert len(details) == 1
        assert details[0]["is_correct"] is True
    
    def test_memory_accuracy_with_poor_match(self, metrics_collector):
        """Test memory accuracy with poor matches."""
        retrieved = [
            {
                "title": "Climate Change",
                "content": "Global warming is a serious environmental issue"
            }
        ]
        
        expected = [
            {
                "title": "AI Technology",
                "content": "Artificial intelligence is transforming industries"
            }
        ]
        
        result = metrics_collector.measure_memory_accuracy(retrieved, expected)
        
        # Should have low accuracy with poor match
        assert result.value < 0.5
        assert not result.passed
        
        # Check retrieval details
        details = result.metadata["retrieval_details"]
        assert len(details) == 1
        assert details[0]["is_correct"] is False
    
    def test_collaboration_efficiency_measurement(self, metrics_collector):
        """Test collaboration efficiency measurement."""
        workflow_log = {
            "handoffs": [
                {"agent_from": "research", "agent_to": "analysis", "success": True},
                {"agent_from": "analysis", "agent_to": "synthesis", "success": True},
                {"agent_from": "synthesis", "agent_to": "validation", "success": False}
            ],
            "communications": [
                {"type": "request", "effective": True},
                {"type": "response", "effective": True},
                {"type": "error", "effective": False}
            ]
        }
        
        results = metrics_collector.measure_collaboration_efficiency(workflow_log)
        
        assert len(results) == 2  # Should return handoff and communication metrics
        
        # Check handoff success rate
        handoff_result = next(r for r in results if r.name == "handoff_success_rate")
        assert handoff_result.value == 2/3  # 2 out of 3 successful
        assert handoff_result.category == "collaboration"
        assert handoff_result.metadata["total_handoffs"] == 3
        assert handoff_result.metadata["successful_handoffs"] == 2
        
        # Check communication efficiency
        comm_result = next(r for r in results if r.name == "communication_efficiency")
        assert comm_result.value == 2/3  # 2 out of 3 effective
        assert comm_result.category == "collaboration"
        assert comm_result.metadata["total_communications"] == 3
        assert comm_result.metadata["effective_communications"] == 2
    
    def test_collaboration_efficiency_with_empty_log(self, metrics_collector):
        """Test collaboration efficiency with empty workflow log."""
        workflow_log = {"handoffs": [], "communications": []}
        
        results = metrics_collector.measure_collaboration_efficiency(workflow_log)
        
        assert len(results) == 2
        
        # Both should have 0 efficiency
        for result in results:
            assert result.value == 0.0
            assert not result.passed
    
    def test_memory_learning_benchmark_execution(self, benchmark_runner):
        """Test that memory learning benchmark can be executed."""
        results = benchmark_runner.run_memory_learning_benchmark()
        
        assert isinstance(results, list)
        assert len(results) > 0, "Should have at least one test result"
        
        # Check that all results are from technology/business categories
        for result in results:
            assert isinstance(result, BenchmarkResult)
            assert result.category in ["Technology", "Business"]
    
    def test_memory_learning_metrics_collection(self, benchmark_runner):
        """Test that memory learning metrics are properly collected."""
        test_case = {
            "id": "test-memory-001",
            "category": "Technology",
            "query": "AI technology trends",
            "expected": {
                "min_memory_precision": 0.7
            }
        }
        
        extracted_metrics = {
            "similar_research": [
                {
                    "title": "Previous AI Research",
                    "content": "Previous study on AI trends",
                    "similarity_score": 0.8
                }
            ],
            "sources": [
                {
                    "title": "Current AI Study",
                    "content": "Current research on AI technology",
                    "url": "https://example.com/ai-study"
                }
            ],
            "execution_results": []
        }
        
        metrics = benchmark_runner._collect_metrics_for_test(test_case, extracted_metrics, 60.0)
        
        # Should have memory precision metric
        metric_names = [m.name for m in metrics]
        assert "memory_precision" in metric_names
        
        # Check memory precision metric
        memory_metric = next(m for m in metrics if m.name == "memory_precision")
        assert memory_metric.category == "memory_learning"
        assert 0 <= memory_metric.value <= 1
    
    def test_memory_learning_threshold_evaluation(self, benchmark_runner):
        """Test memory learning threshold evaluation."""
        test_case = {
            "id": "test-memory-threshold",
            "category": "Technology",
            "query": "Technology trends",
            "expected": {
                "min_memory_precision": 0.7
            }
        }
        
        # Create metrics that should pass
        metrics = [
            type('Metric', (), {
                'name': 'memory_precision',
                'value': 0.8,
                'threshold': 0.7,
                'passed': True,
                'category': 'memory_learning'
            })()
        ]
        
        passed = benchmark_runner._evaluate_test_pass(metrics, test_case['expected'])
        assert passed, "Test should pass with good memory metrics"
        
        # Create metrics that should fail
        failing_metrics = [
            type('Metric', (), {
                'name': 'memory_precision',
                'value': 0.5,
                'threshold': 0.7,
                'passed': False,
                'category': 'memory_learning'
            })()
        ]
        
        failed = benchmark_runner._evaluate_test_pass(failing_metrics, test_case['expected'])
        assert not failed, "Test should fail with poor memory metrics"
    
    def test_similarity_calculation_accuracy(self, metrics_collector):
        """Test text similarity calculation accuracy."""
        # Test identical texts
        identical_similarity = metrics_collector._calculate_similarity(
            "AI is advancing rapidly",
            "AI is advancing rapidly"
        )
        assert identical_similarity == 1.0
        
        # Test completely different texts
        different_similarity = metrics_collector._calculate_similarity(
            "AI technology trends",
            "Climate change research"
        )
        assert different_similarity < 0.5
        
        # Test partially similar texts
        partial_similarity = metrics_collector._calculate_similarity(
            "AI is advancing rapidly in technology",
            "AI technology is advancing quickly"
        )
        assert 0.3 < partial_similarity < 0.9
    
    def test_memory_learning_benchmark_summary(self, benchmark_runner):
        """Test memory learning benchmark summary generation."""
        mock_results = [
            BenchmarkResult(
                test_id="memory-001",
                category="Technology",
                query="AI trends",
                execution_time=100.0,
                metrics=[],
                overall_score=0.8,
                passed=True,
                timestamp=None
            ),
            BenchmarkResult(
                test_id="memory-002",
                category="Business",
                query="Business innovation",
                execution_time=120.0,
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
        assert "Technology" in breakdown
        assert "Business" in breakdown
    
    def test_memory_retrieval_details(self, metrics_collector):
        """Test memory retrieval details in metadata."""
        retrieved = [
            {
                "title": "AI Research 1",
                "content": "Study on artificial intelligence"
            },
            {
                "title": "ML Study",
                "content": "Machine learning research"
            }
        ]
        
        expected = [
            {
                "title": "AI Research 2",
                "content": "Research on artificial intelligence"
            }
        ]
        
        result = metrics_collector.measure_memory_accuracy(retrieved, expected)
        
        # Check retrieval details
        details = result.metadata["retrieval_details"]
        assert len(details) == 2
        
        for detail in details:
            assert "retrieved_title" in detail
            assert "is_correct" in detail
            assert "similarity_scores" in detail
            assert "title" in detail["similarity_scores"]
            assert "content" in detail["similarity_scores"]
    
    def test_collaboration_handoff_details(self, metrics_collector):
        """Test collaboration handoff details in metadata."""
        workflow_log = {
            "handoffs": [
                {"agent_from": "research", "agent_to": "analysis", "success": True, "timestamp": "2025-01-01T10:00:00Z"},
                {"agent_from": "analysis", "agent_to": "synthesis", "success": False, "error": "Timeout"}
            ],
            "communications": []
        }
        
        results = metrics_collector.measure_collaboration_efficiency(workflow_log)
        
        handoff_result = next(r for r in results if r.name == "handoff_success_rate")
        handoff_details = handoff_result.metadata["handoff_details"]
        
        assert len(handoff_details) == 2
        assert handoff_details[0]["success"] is True
        assert handoff_details[1]["success"] is False
        assert "error" in handoff_details[1]
    
    @pytest.mark.slow
    def test_full_memory_learning_benchmark(self, benchmark_runner):
        """Test full memory learning benchmark execution (slow test)."""
        # This test actually runs the CLI and may take time
        results = benchmark_runner.run_memory_learning_benchmark()
        
        assert len(results) > 0, "Should have results from memory learning benchmark"
        
        # Check that we have results from memory-focused categories
        categories = {result.category for result in results}
        expected_categories = {"Technology", "Business"}
        assert categories.intersection(expected_categories), "Should have results from memory categories"
    
    def test_memory_learning_benchmark_configuration(self, benchmark_runner):
        """Test that memory learning benchmark uses correct configuration."""
        # Check that the benchmark runner has the right configuration
        assert benchmark_runner.config is not None
        assert "test_cases" in benchmark_runner.config
        
        # Check that technology/business test cases exist
        test_cases = benchmark_runner.config["test_cases"]
        memory_cases = [tc for tc in test_cases if tc.get("category") in ["Technology", "Business"]]
        assert len(memory_cases) > 0, "Should have memory learning test cases in configuration"
    
    def test_benchmark_result_creation_with_memory_metrics(self):
        """Test BenchmarkResult creation with memory learning metrics."""
        result = BenchmarkResult(
            test_id="memory-test-001",
            category="Technology",
            query="AI memory test",
            execution_time=90.0,
            metrics=[],
            overall_score=0.75,
            passed=True,
            timestamp=None
        )
        
        assert result.test_id == "memory-test-001"
        assert result.category == "Technology"
        assert result.query == "AI memory test"
        assert result.execution_time == 90.0
        assert result.overall_score == 0.75
        assert result.passed is True
        assert isinstance(result.metrics, list)
