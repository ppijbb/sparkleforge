"""
SparkleForge Agent Evaluation Metrics Collection and Analysis

This module provides comprehensive agent evaluation metrics based on academic standards
(WebArena, ToolBench, AgentBench, ALFWorld) for evaluating SparkleForge agent performance.
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
class AgentMetricResult:
    """Individual agent metric result with value and metadata."""
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
class AgentBenchmarkResult:
    """Complete agent benchmark result for a single task."""
    task_id: str
    category: str
    task_type: str
    query: str
    execution_time: float
    metrics: List[AgentMetricResult]
    overall_score: float
    passed: bool
    timestamp: datetime
    raw_output: Dict[str, Any] = None

    def __post_init__(self):
        if self.raw_output is None:
            self.raw_output = {}


class AgentMetricsCollector:
    """Collects and analyzes agent performance metrics based on academic standards."""
    
    def __init__(self, thresholds: Dict[str, float]):
        self.thresholds = thresholds
        self.logger = logging.getLogger(__name__)
    
    # Web Navigation Metrics (WebArena-style)
    def measure_navigation_success(self, navigation_events: List[Dict]) -> AgentMetricResult:
        """Measure web navigation success rate."""
        if not navigation_events:
            return AgentMetricResult(
                name="navigation_success_rate",
                value=0.0,
                threshold=self.thresholds.get('navigation_success_rate', 0.8),
                passed=False,
                category="web_navigation",
                metadata={"total_events": 0, "successful_events": 0}
            )
        
        successful_navigations = sum(1 for event in navigation_events 
                                   if event.get('status') == 'success')
        success_rate = successful_navigations / len(navigation_events)
        threshold = self.thresholds.get('navigation_success_rate', 0.8)
        
        return AgentMetricResult(
            name="navigation_success_rate",
            value=success_rate,
            threshold=threshold,
            passed=success_rate >= threshold,
            category="web_navigation",
            metadata={
                "total_events": len(navigation_events),
                "successful_events": successful_navigations,
                "failed_events": len(navigation_events) - successful_navigations
            }
        )
    
    def measure_information_accuracy(self, retrieved_info: List[Dict], expected_info: List[str]) -> AgentMetricResult:
        """Measure accuracy of retrieved information."""
        if not retrieved_info or not expected_info:
            return AgentMetricResult(
                name="information_accuracy",
                value=0.0,
                threshold=self.thresholds.get('information_accuracy', 0.85),
                passed=False,
                category="web_navigation",
                metadata={"retrieved_count": 0, "expected_count": 0}
            )
        
        # Simple keyword matching for accuracy measurement
        retrieved_text = " ".join([str(item.get('content', '')) for item in retrieved_info]).lower()
        expected_keywords = [keyword.lower() for keyword in expected_info]
        
        matched_keywords = sum(1 for keyword in expected_keywords 
                             if keyword in retrieved_text)
        accuracy = matched_keywords / len(expected_keywords) if expected_keywords else 0.0
        threshold = self.thresholds.get('information_accuracy', 0.85)
        
        return AgentMetricResult(
            name="information_accuracy",
            value=accuracy,
            threshold=threshold,
            passed=accuracy >= threshold,
            category="web_navigation",
            metadata={
                "retrieved_count": len(retrieved_info),
                "expected_keywords": len(expected_keywords),
                "matched_keywords": matched_keywords
            }
        )
    
    # Tool Usage Metrics (ToolBench-style)
    def measure_tool_usage_success(self, tool_events: List[Dict]) -> AgentMetricResult:
        """Measure tool usage success rate."""
        if not tool_events:
            return AgentMetricResult(
                name="tool_usage_success_rate",
                value=0.0,
                threshold=self.thresholds.get('tool_usage_success_rate', 0.85),
                passed=False,
                category="tool_usage",
                metadata={"total_tools": 0, "successful_tools": 0}
            )
        
        successful_tools = sum(1 for event in tool_events 
                             if event.get('status') == 'success')
        success_rate = successful_tools / len(tool_events)
        threshold = self.thresholds.get('tool_usage_success_rate', 0.85)
        
        return AgentMetricResult(
            name="tool_usage_success_rate",
            value=success_rate,
            threshold=threshold,
            passed=success_rate >= threshold,
            category="tool_usage",
            metadata={
                "total_tools": len(tool_events),
                "successful_tools": successful_tools,
                "failed_tools": len(tool_events) - successful_tools
            }
        )
    
    def measure_tool_coordination_efficiency(self, coordination_events: List[Dict]) -> AgentMetricResult:
        """Measure efficiency of tool coordination."""
        if not coordination_events:
            return AgentMetricResult(
                name="tool_coordination_efficiency",
                value=0.0,
                threshold=self.thresholds.get('tool_coordination_efficiency', 0.8),
                passed=False,
                category="tool_usage",
                metadata={"coordination_events": 0}
            )
        
        # Calculate efficiency based on coordination success and timing
        successful_coordinations = sum(1 for event in coordination_events 
                                     if event.get('coordination_success', False))
        avg_coordination_time = statistics.mean([event.get('coordination_time', 0) 
                                               for event in coordination_events])
        
        # Efficiency = success_rate * (1 - normalized_time)
        success_rate = successful_coordinations / len(coordination_events)
        normalized_time = min(avg_coordination_time / 60.0, 1.0)  # Normalize to 60 seconds
        efficiency = success_rate * (1 - normalized_time)
        
        threshold = self.thresholds.get('tool_coordination_efficiency', 0.8)
        
        return AgentMetricResult(
            name="tool_coordination_efficiency",
            value=efficiency,
            threshold=threshold,
            passed=efficiency >= threshold,
            category="tool_usage",
            metadata={
                "coordination_events": len(coordination_events),
                "successful_coordinations": successful_coordinations,
                "avg_coordination_time": avg_coordination_time
            }
        )
    
    # Multi-Agent Collaboration Metrics (AgentBench-style)
    def measure_coordination_efficiency(self, agent_events: List[Dict]) -> AgentMetricResult:
        """Measure multi-agent coordination efficiency."""
        if not agent_events:
            return AgentMetricResult(
                name="coordination_efficiency",
                value=0.0,
                threshold=self.thresholds.get('coordination_efficiency', 0.8),
                passed=False,
                category="multi_agent",
                metadata={"agent_events": 0}
            )
        
        # Calculate coordination efficiency based on agent interactions
        successful_interactions = sum(1 for event in agent_events 
                                    if event.get('interaction_success', False))
        total_interactions = len(agent_events)
        
        # Factor in communication overhead
        communication_overhead = sum(event.get('communication_time', 0) 
                                   for event in agent_events) / total_interactions
        
        efficiency = (successful_interactions / total_interactions) * (1 - min(communication_overhead / 10.0, 0.5))
        threshold = self.thresholds.get('coordination_efficiency', 0.8)
        
        return AgentMetricResult(
            name="coordination_efficiency",
            value=efficiency,
            threshold=threshold,
            passed=efficiency >= threshold,
            category="multi_agent",
            metadata={
                "agent_events": total_interactions,
                "successful_interactions": successful_interactions,
                "communication_overhead": communication_overhead
            }
        )
    
    def measure_task_completion_rate(self, task_events: List[Dict]) -> AgentMetricResult:
        """Measure task completion rate across agents."""
        if not task_events:
            return AgentMetricResult(
                name="task_completion_rate",
                value=0.0,
                threshold=self.thresholds.get('task_completion_rate', 0.9),
                passed=False,
                category="multi_agent",
                metadata={"total_tasks": 0, "completed_tasks": 0}
            )
        
        completed_tasks = sum(1 for event in task_events 
                             if event.get('task_status') == 'completed')
        completion_rate = completed_tasks / len(task_events)
        threshold = self.thresholds.get('task_completion_rate', 0.9)
        
        return AgentMetricResult(
            name="task_completion_rate",
            value=completion_rate,
            threshold=threshold,
            passed=completion_rate >= threshold,
            category="multi_agent",
            metadata={
                "total_tasks": len(task_events),
                "completed_tasks": completed_tasks,
                "failed_tasks": len(task_events) - completed_tasks
            }
        )
    
    # Reasoning and Planning Metrics (ALFWorld-style)
    def measure_reasoning_accuracy(self, reasoning_steps: List[Dict]) -> AgentMetricResult:
        """Measure accuracy of logical reasoning steps."""
        if not reasoning_steps:
            return AgentMetricResult(
                name="reasoning_accuracy",
                value=0.0,
                threshold=self.thresholds.get('reasoning_accuracy', 0.9),
                passed=False,
                category="reasoning",
                metadata={"reasoning_steps": 0, "valid_steps": 0}
            )
        
        valid_steps = sum(1 for step in reasoning_steps 
                         if step.get('logical_validity', False))
        accuracy = valid_steps / len(reasoning_steps)
        threshold = self.thresholds.get('reasoning_accuracy', 0.9)
        
        return AgentMetricResult(
            name="reasoning_accuracy",
            value=accuracy,
            threshold=threshold,
            passed=accuracy >= threshold,
            category="reasoning",
            metadata={
                "reasoning_steps": len(reasoning_steps),
                "valid_steps": valid_steps,
                "invalid_steps": len(reasoning_steps) - valid_steps
            }
        )
    
    def measure_plan_feasibility(self, plan_steps: List[Dict]) -> AgentMetricResult:
        """Measure feasibility of generated plans."""
        if not plan_steps:
            return AgentMetricResult(
                name="plan_feasibility",
                value=0.0,
                threshold=self.thresholds.get('plan_feasibility', 0.9),
                passed=False,
                category="reasoning",
                metadata={"plan_steps": 0, "feasible_steps": 0}
            )
        
        feasible_steps = sum(1 for step in plan_steps 
                           if step.get('feasibility_score', 0) >= 0.7)
        feasibility = feasible_steps / len(plan_steps)
        threshold = self.thresholds.get('plan_feasibility', 0.9)
        
        return AgentMetricResult(
            name="plan_feasibility",
            value=feasibility,
            threshold=threshold,
            passed=feasibility >= threshold,
            category="reasoning",
            metadata={
                "plan_steps": len(plan_steps),
                "feasible_steps": feasible_steps,
                "infeasible_steps": len(plan_steps) - feasible_steps
            }
        )
    
    # Overall Agent Performance Metrics
    def measure_execution_efficiency(self, execution_events: List[Dict]) -> AgentMetricResult:
        """Measure overall execution efficiency."""
        if not execution_events:
            return AgentMetricResult(
                name="execution_efficiency",
                value=0.0,
                threshold=self.thresholds.get('execution_efficiency', 0.8),
                passed=False,
                category="agent_performance",
                metadata={"execution_events": 0}
            )
        
        # Calculate efficiency based on successful executions and resource usage
        successful_executions = sum(1 for event in execution_events 
                                  if event.get('execution_success', False))
        avg_resource_usage = statistics.mean([event.get('resource_usage', 1.0) 
                                            for event in execution_events])
        
        efficiency = (successful_executions / len(execution_events)) * (1 - min(avg_resource_usage, 1.0))
        threshold = self.thresholds.get('execution_efficiency', 0.8)
        
        return AgentMetricResult(
            name="execution_efficiency",
            value=efficiency,
            threshold=threshold,
            passed=efficiency >= threshold,
            category="agent_performance",
            metadata={
                "execution_events": len(execution_events),
                "successful_executions": successful_executions,
                "avg_resource_usage": avg_resource_usage
            }
        )
    
    def measure_reliability_score(self, reliability_events: List[Dict]) -> AgentMetricResult:
        """Measure system reliability score."""
        if not reliability_events:
            return AgentMetricResult(
                name="reliability_score",
                value=0.0,
                threshold=self.thresholds.get('reliability_score', 0.9),
                passed=False,
                category="agent_performance",
                metadata={"reliability_events": 0}
            )
        
        # Calculate reliability based on uptime and error rates
        uptime_events = sum(1 for event in reliability_events 
                           if event.get('status') == 'up')
        error_events = sum(1 for event in reliability_events 
                          if event.get('error_type') is not None)
        
        uptime_rate = uptime_events / len(reliability_events)
        error_rate = error_events / len(reliability_events)
        reliability = uptime_rate * (1 - error_rate)
        
        threshold = self.thresholds.get('reliability_score', 0.9)
        
        return AgentMetricResult(
            name="reliability_score",
            value=reliability,
            threshold=threshold,
            passed=reliability >= threshold,
            category="agent_performance",
            metadata={
                "reliability_events": len(reliability_events),
                "uptime_events": uptime_events,
                "error_events": error_events,
                "uptime_rate": uptime_rate,
                "error_rate": error_rate
            }
        )


class AgentBenchmarkAnalyzer:
    """Analyzes agent benchmark results and provides insights."""
    
    def __init__(self, thresholds: Dict[str, float], weights: Dict[str, float] = None):
        self.thresholds = thresholds
        self.weights = weights or {
            'web_navigation': 0.25,
            'tool_usage': 0.25,
            'multi_agent': 0.25,
            'reasoning': 0.15,
            'agent_performance': 0.10
        }
        self.logger = logging.getLogger(__name__)
    
    def calculate_overall_score(self, results: List[AgentBenchmarkResult]) -> float:
        """Calculate weighted overall score from all benchmark results."""
        if not results:
            return 0.0
        
        # Use the overall_score from each result directly
        scores = [result.overall_score for result in results if result.overall_score is not None]
        
        if not scores:
            return 0.0
        
        return statistics.mean(scores)
    
    def detect_regressions(self, current_results: List[AgentBenchmarkResult], 
                          baseline_results: List[AgentBenchmarkResult]) -> List[Dict]:
        """Detect performance regressions compared to baseline."""
        regressions = []
        
        if not baseline_results:
            return regressions
        
        # Create baseline lookup
        baseline_lookup = {result.task_id: result for result in baseline_results}
        
        for current_result in current_results:
            if current_result.task_id in baseline_lookup:
                baseline_result = baseline_lookup[current_result.task_id]
                
                if current_result.overall_score < baseline_result.overall_score * 0.95:  # 5% regression threshold
                    regressions.append({
                        'task_id': current_result.task_id,
                        'category': current_result.category,
                        'current_score': current_result.overall_score,
                        'baseline_score': baseline_result.overall_score,
                        'regression_percentage': ((baseline_result.overall_score - current_result.overall_score) / baseline_result.overall_score) * 100
                    })
        
        return regressions
    
    def generate_improvement_report(self, current_results: List[AgentBenchmarkResult], 
                                   baseline_results: List[AgentBenchmarkResult]) -> Dict:
        """Generate improvement report comparing current vs baseline results."""
        if not baseline_results:
            return {"message": "No baseline results available for comparison"}
        
        current_score = self.calculate_overall_score(current_results)
        baseline_score = self.calculate_overall_score(baseline_results)
        
        improvement = current_score - baseline_score
        improvement_percentage = (improvement / baseline_score) * 100 if baseline_score > 0 else 0
        
        # Category-wise analysis
        category_analysis = {}
        for category in ['web_navigation', 'tool_usage', 'multi_agent', 'reasoning', 'agent_performance']:
            current_category_results = [r for r in current_results 
                                      if any(m.category == category for m in r.metrics)]
            baseline_category_results = [r for r in baseline_results 
                                       if any(m.category == category for m in r.metrics)]
            
            if current_category_results and baseline_category_results:
                current_category_score = statistics.mean([r.overall_score for r in current_category_results])
                baseline_category_score = statistics.mean([r.overall_score for r in baseline_category_results])
                
                category_analysis[category] = {
                    'current_score': current_category_score,
                    'baseline_score': baseline_category_score,
                    'improvement': current_category_score - baseline_category_score,
                    'improvement_percentage': ((current_category_score - baseline_category_score) / baseline_category_score) * 100 if baseline_category_score > 0 else 0
                }
        
        return {
            'overall_improvement': improvement,
            'overall_improvement_percentage': improvement_percentage,
            'current_overall_score': current_score,
            'baseline_overall_score': baseline_score,
            'category_analysis': category_analysis,
            'regressions_detected': len(self.detect_regressions(current_results, baseline_results))
        }
