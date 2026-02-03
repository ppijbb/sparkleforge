"""
SparkleForge Benchmark Metrics Collection and Analysis

This module provides comprehensive metrics collection and analysis for
evaluating SparkleForge performance across all benchmark categories.
"""

import time
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import statistics
import re

logger = logging.getLogger(__name__)


@dataclass
class MetricResult:
    """Individual metric result with value and metadata."""
    name: str
    value: float
    threshold: float
    passed: bool
    category: str
    weight: float = 1.0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class BenchmarkResult:
    """Complete benchmark result for a single test case."""
    test_id: str
    category: str
    query: str
    execution_time: float
    metrics: List[MetricResult]
    overall_score: float
    passed: bool
    timestamp: datetime
    raw_output: Dict[str, Any] = None

    def __post_init__(self):
        if self.raw_output is None:
            self.raw_output = {}


class MetricsCollector:
    """Collects and analyzes various performance metrics."""
    
    def __init__(self, thresholds: Dict[str, float]):
        self.thresholds = thresholds
        self.logger = logging.getLogger(__name__)
    
    def measure_response_time(self, start_time: float, end_time: float) -> MetricResult:
        """Measure response time and compare against threshold."""
        response_time = end_time - start_time
        threshold = self.thresholds.get('response_time', 120.0)
        
        return MetricResult(
            name="response_time",
            value=response_time,
            threshold=threshold,
            passed=response_time <= threshold,
            category="performance",
            metadata={
                "start_time": start_time,
                "end_time": end_time,
                "formatted_time": f"{response_time:.2f}s"
            }
        )
    
    def measure_source_credibility(self, sources: List[Dict[str, Any]]) -> MetricResult:
        """Measure average source credibility score."""
        if not sources:
            return MetricResult(
                name="source_credibility",
                value=0.0,
                threshold=self.thresholds.get('source_credibility', 0.7),
                passed=False,
                category="research_quality",
                metadata={"error": "No sources found"}
            )
        
        credibility_scores = []
        for source in sources:
            # Extract credibility score from source metadata
            cred_score = source.get('credibility_score', 0.0)
            if isinstance(cred_score, (int, float)):
                credibility_scores.append(cred_score)
            else:
                # Try to extract from domain or other metadata
                domain = source.get('domain', '')
                cred_score = self._estimate_domain_credibility(domain)
                credibility_scores.append(cred_score)
        
        avg_credibility = statistics.mean(credibility_scores) if credibility_scores else 0.0
        threshold = self.thresholds.get('source_credibility', 0.7)
        
        return MetricResult(
            name="source_credibility",
            value=avg_credibility,
            threshold=threshold,
            passed=avg_credibility >= threshold,
            category="research_quality",
            metadata={
                "source_count": len(sources),
                "credibility_scores": credibility_scores,
                "min_credibility": min(credibility_scores) if credibility_scores else 0,
                "max_credibility": max(credibility_scores) if credibility_scores else 0
            }
        )
    
    def measure_factual_accuracy(self, claims: List[str], sources: List[Dict[str, Any]]) -> MetricResult:
        """Measure factual accuracy by checking claims against sources."""
        if not claims or not sources:
            return MetricResult(
                name="factual_accuracy",
                value=0.0,
                threshold=self.thresholds.get('factual_accuracy', 0.85),
                passed=False,
                category="research_quality",
                metadata={"error": "No claims or sources to verify"}
            )
        
        # Simple keyword-based verification (in production, use more sophisticated NLP)
        verified_claims = 0
        verification_details = []
        
        for claim in claims:
            claim_verified = False
            supporting_sources = []
            
            # Extract keywords from claim
            claim_keywords = self._extract_keywords(claim.lower())
            
            for source in sources:
                source_text = source.get('content', '').lower()
                source_title = source.get('title', '').lower()
                
                # Check if claim keywords appear in source
                keyword_matches = sum(1 for kw in claim_keywords if kw in source_text or kw in source_title)
                if keyword_matches >= len(claim_keywords) * 0.5:  # At least 50% keyword match
                    claim_verified = True
                    supporting_sources.append(source.get('url', 'unknown'))
            
            if claim_verified:
                verified_claims += 1
            
            verification_details.append({
                "claim": claim,
                "verified": claim_verified,
                "supporting_sources": supporting_sources
            })
        
        accuracy = verified_claims / len(claims) if claims else 0.0
        threshold = self.thresholds.get('factual_accuracy', 0.85)
        
        return MetricResult(
            name="factual_accuracy",
            value=accuracy,
            threshold=threshold,
            passed=accuracy >= threshold,
            category="research_quality",
            metadata={
                "total_claims": len(claims),
                "verified_claims": verified_claims,
                "verification_details": verification_details
            }
        )
    
    def measure_creative_quality(self, insights: List[Dict[str, Any]]) -> List[MetricResult]:
        """Measure creative insights quality metrics."""
        if not insights:
            return [
                MetricResult(
                    name="creative_novelty",
                    value=0.0,
                    threshold=self.thresholds.get('creative_novelty', 0.6),
                    passed=False,
                    category="creative_insights",
                    metadata={"error": "No insights found"}
                ),
                MetricResult(
                    name="creative_applicability",
                    value=0.0,
                    threshold=self.thresholds.get('creative_applicability', 0.6),
                    passed=False,
                    category="creative_insights",
                    metadata={"error": "No insights found"}
                )
            ]
        
        novelty_scores = []
        applicability_scores = []
        insight_details = []
        
        for insight in insights:
            novelty = insight.get('novelty_score', 0.0)
            applicability = insight.get('applicability_score', 0.0)
            
            novelty_scores.append(novelty)
            applicability_scores.append(applicability)
            
            insight_details.append({
                "title": insight.get('title', 'Unknown'),
                "type": insight.get('type', 'unknown'),
                "novelty_score": novelty,
                "applicability_score": applicability,
                "confidence": insight.get('confidence', 0.0)
            })
        
        avg_novelty = statistics.mean(novelty_scores) if novelty_scores else 0.0
        avg_applicability = statistics.mean(applicability_scores) if applicability_scores else 0.0
        
        return [
            MetricResult(
                name="creative_novelty",
                value=avg_novelty,
                threshold=self.thresholds.get('creative_novelty', 0.6),
                passed=avg_novelty >= self.thresholds.get('creative_novelty', 0.6),
                category="creative_insights",
                metadata={
                    "insight_count": len(insights),
                    "novelty_scores": novelty_scores,
                    "insight_details": insight_details
                }
            ),
            MetricResult(
                name="creative_applicability",
                value=avg_applicability,
                threshold=self.thresholds.get('creative_applicability', 0.6),
                passed=avg_applicability >= self.thresholds.get('creative_applicability', 0.6),
                category="creative_insights",
                metadata={
                    "insight_count": len(insights),
                    "applicability_scores": applicability_scores,
                    "insight_details": insight_details
                }
            )
        ]
    
    def measure_memory_accuracy(self, retrieved: List[Dict[str, Any]], expected: List[Dict[str, Any]]) -> MetricResult:
        """Measure memory retrieval accuracy."""
        if not retrieved:
            return MetricResult(
                name="memory_precision",
                value=0.0,
                threshold=self.thresholds.get('memory_precision', 0.7),
                passed=False,
                category="memory_learning",
                metadata={"error": "No retrieved memories"}
            )
        
        # Simple similarity matching (in production, use vector similarity)
        correct_retrievals = 0
        retrieval_details = []
        
        for ret_item in retrieved:
            ret_title = ret_item.get('title', '').lower()
            ret_content = ret_item.get('content', '').lower()
            
            is_correct = False
            for exp_item in expected:
                exp_title = exp_item.get('title', '').lower()
                exp_content = exp_item.get('content', '').lower()
                
                # Check title similarity
                title_similarity = self._calculate_similarity(ret_title, exp_title)
                content_similarity = self._calculate_similarity(ret_content, exp_content)
                
                if title_similarity > 0.7 or content_similarity > 0.6:
                    is_correct = True
                    break
            
            if is_correct:
                correct_retrievals += 1
            
            retrieval_details.append({
                "retrieved_title": ret_item.get('title', ''),
                "is_correct": is_correct,
                "similarity_scores": {
                    "title": title_similarity if 'title_similarity' in locals() else 0,
                    "content": content_similarity if 'content_similarity' in locals() else 0
                }
            })
        
        precision = correct_retrievals / len(retrieved) if retrieved else 0.0
        threshold = self.thresholds.get('memory_precision', 0.7)
        
        return MetricResult(
            name="memory_precision",
            value=precision,
            threshold=threshold,
            passed=precision >= threshold,
            category="memory_learning",
            metadata={
                "total_retrieved": len(retrieved),
                "correct_retrievals": correct_retrievals,
                "retrieval_details": retrieval_details
            }
        )
    
    def measure_collaboration_efficiency(self, workflow_log: Dict[str, Any]) -> List[MetricResult]:
        """Measure agent collaboration efficiency."""
        handoffs = workflow_log.get('handoffs', [])
        communications = workflow_log.get('communications', [])
        
        # Calculate handoff success rate
        successful_handoffs = sum(1 for h in handoffs if h.get('success', False))
        handoff_rate = successful_handoffs / len(handoffs) if handoffs else 0.0
        
        # Calculate communication efficiency (simplified)
        total_communications = len(communications)
        effective_communications = sum(1 for c in communications if c.get('effective', True))
        comm_efficiency = effective_communications / total_communications if total_communications else 0.0
        
        return [
            MetricResult(
                name="handoff_success_rate",
                value=handoff_rate,
                threshold=self.thresholds.get('handoff_success_rate', 0.9),
                passed=handoff_rate >= self.thresholds.get('handoff_success_rate', 0.9),
                category="collaboration",
                metadata={
                    "total_handoffs": len(handoffs),
                    "successful_handoffs": successful_handoffs,
                    "handoff_details": handoffs
                }
            ),
            MetricResult(
                name="communication_efficiency",
                value=comm_efficiency,
                threshold=self.thresholds.get('communication_efficiency', 0.8),
                passed=comm_efficiency >= self.thresholds.get('communication_efficiency', 0.8),
                category="collaboration",
                metadata={
                    "total_communications": total_communications,
                    "effective_communications": effective_communications,
                    "communication_details": communications
                }
            )
        ]
    
    def _estimate_domain_credibility(self, domain: str) -> float:
        """Estimate credibility based on domain reputation."""
        if not domain:
            return 0.5
        
        # Simple domain-based credibility estimation
        high_cred_domains = ['.edu', '.gov', '.org', 'nature.com', 'science.org', 'arxiv.org']
        medium_cred_domains = ['.com', '.net', 'wikipedia.org', 'medium.com']
        
        domain_lower = domain.lower()
        
        if any(high_domain in domain_lower for high_domain in high_cred_domains):
            return 0.9
        elif any(med_domain in domain_lower for med_domain in medium_cred_domains):
            return 0.7
        else:
            return 0.5
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text."""
        # Simple keyword extraction (in production, use NLP libraries)
        words = re.findall(r'\b\w+\b', text.lower())
        # Filter out common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'}
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        return keywords[:10]  # Return top 10 keywords
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity."""
        if not text1 or not text2:
            return 0.0
        
        words1 = set(self._extract_keywords(text1))
        words2 = set(self._extract_keywords(text2))
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0


class BenchmarkAnalyzer:
    """Analyzes benchmark results and generates insights."""
    
    def __init__(self, thresholds: Dict[str, float], weights: Dict[str, float]):
        self.thresholds = thresholds
        self.weights = weights
        self.logger = logging.getLogger(__name__)
    
    def calculate_overall_score(self, results: List[BenchmarkResult]) -> float:
        """Calculate weighted overall score from all benchmark results."""
        if not results:
            return 0.0
        
        # Use the overall_score from each result directly
        scores = [result.overall_score for result in results if result.overall_score is not None]
        
        if not scores:
            return 0.0
        
        return statistics.mean(scores)
    
    def detect_regressions(self, current_results: List[BenchmarkResult], 
                          previous_results: List[BenchmarkResult]) -> List[Dict[str, Any]]:
        """Detect performance regressions compared to previous results."""
        regressions = []
        
        if not previous_results:
            return regressions
        
        # Create lookup for previous results
        prev_lookup = {r.test_id: r for r in previous_results}
        
        for current in current_results:
            if current.test_id not in prev_lookup:
                continue
            
            prev = prev_lookup[current.test_id]
            
            # Check for performance degradation
            if current.execution_time > prev.execution_time * 1.1:  # 10% slower
                regressions.append({
                    "test_id": current.test_id,
                    "type": "performance_degradation",
                    "current_value": current.execution_time,
                    "previous_value": prev.execution_time,
                    "degradation_percent": ((current.execution_time - prev.execution_time) / prev.execution_time) * 100
                })
            
            # Check for quality degradation
            if current.overall_score < prev.overall_score * 0.95:  # 5% lower quality
                regressions.append({
                    "test_id": current.test_id,
                    "type": "quality_degradation",
                    "current_value": current.overall_score,
                    "previous_value": prev.overall_score,
                    "degradation_percent": ((prev.overall_score - current.overall_score) / prev.overall_score) * 100
                })
        
        return regressions
    
    def generate_insights(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Generate insights and recommendations from benchmark results."""
        if not results:
            return {"error": "No results to analyze"}
        
        insights = {
            "summary": {
                "total_tests": len(results),
                "passed_tests": sum(1 for r in results if r.passed),
                "failed_tests": sum(1 for r in results if not r.passed),
                "overall_score": self.calculate_overall_score(results)
            },
            "category_analysis": {},
            "recommendations": []
        }
        
        # Analyze by category
        category_metrics = {}
        for result in results:
            for metric in result.metrics:
                category = metric.category
                if category not in category_metrics:
                    category_metrics[category] = []
                category_metrics[category].append(metric)
        
        for category, metrics in category_metrics.items():
            passed_count = sum(1 for m in metrics if m.passed)
            total_count = len(metrics)
            avg_score = statistics.mean([m.value for m in metrics])
            
            insights["category_analysis"][category] = {
                "passed_metrics": passed_count,
                "total_metrics": total_count,
                "pass_rate": passed_count / total_count if total_count > 0 else 0,
                "average_score": avg_score,
                "threshold": self.thresholds.get(category, 0.7)
            }
        
        # Generate recommendations
        for category, analysis in insights["category_analysis"].items():
            if analysis["pass_rate"] < 0.8:
                insights["recommendations"].append({
                    "category": category,
                    "issue": f"Low pass rate: {analysis['pass_rate']:.1%}",
                    "recommendation": f"Focus on improving {category} metrics"
                })
        
        return insights
