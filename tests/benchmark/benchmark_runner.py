"""
SparkleForge Benchmark Runner

Main benchmark orchestration class that coordinates all benchmark testing.
Loads configurations, executes tests via CLI, collects metrics, and generates reports.
"""

import yaml
import json
import time
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from .benchmark_metrics import MetricsCollector, BenchmarkAnalyzer, BenchmarkResult, MetricResult
    from .cli_executor import CLIExecutor
except ImportError:
    from benchmark_metrics import MetricsCollector, BenchmarkAnalyzer, BenchmarkResult, MetricResult
    from cli_executor import CLIExecutor

logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """Main benchmark orchestration class."""
    
    def __init__(self, project_root: str, config_path: str, thresholds_path: str):
        self.project_root = Path(project_root)
        self.config_path = Path(config_path)
        self.thresholds_path = Path(thresholds_path)
        
        # Load configurations
        self.config = self._load_config()
        self.thresholds = self._load_thresholds()
        self.weights = self.thresholds.get('scoring_weights', {})
        
        # Initialize components
        self.metrics_collector = MetricsCollector(self.thresholds)
        self.analyzer = BenchmarkAnalyzer(self.thresholds, self.weights)
        self.cli_executor = CLIExecutor(str(self.project_root), self.config.get('execution', {}).get('timeout', 300))
        
        self.logger = logging.getLogger(__name__)
    
    def _load_config(self) -> Dict[str, Any]:
        """Load benchmark configuration."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            self.logger.error(f"Failed to load config: {e}")
            return {}
    
    def _load_thresholds(self) -> Dict[str, Any]:
        """Load benchmark thresholds."""
        try:
            with open(self.thresholds_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            self.logger.error(f"Failed to load thresholds: {e}")
            return {}
    
    def run_all_benchmarks(self) -> List[BenchmarkResult]:
        """Run comprehensive agent evaluation benchmark - all metrics in single execution."""
        self.logger.info("Starting comprehensive agent evaluation benchmark - measuring ALL agent metrics")
        
        # Validate environment first
        is_valid, issues = self.cli_executor.validate_environment()
        if not is_valid:
            self.logger.error(f"Environment validation failed: {issues}")
            return []
        
        agent_tasks = self.config.get('agent_tasks', [])
        if not agent_tasks:
            self.logger.error("No agent tasks found in configuration")
            return []
        
        # Run comprehensive agent benchmarks (all metrics in single execution)
        results = []
        for agent_task in agent_tasks:
            self.logger.info(f"Running agent evaluation for: {agent_task['id']} - {agent_task['task_type']}")
            result = self._run_single_agent_task_comprehensive(agent_task)
            results.append(result)
        
        self.logger.info(f"Completed agent evaluation benchmark: {len(results)} tasks with ALL agent metrics")
        return results
    
    def _run_single_agent_task_comprehensive(self, agent_task: Dict[str, Any]) -> BenchmarkResult:
        """Run a single agent task with comprehensive agent evaluation metrics."""
        start_time = time.time()
        
        try:
            # Execute CLI command for agent task
            cli_result = self.cli_executor.execute_research(agent_task['query'])
            
            if not cli_result.success:
                return BenchmarkResult(
                    test_id=agent_task['id'],
                    category=agent_task.get('category', 'Unknown'),
                    query=agent_task['query'],
                    execution_time=time.time() - start_time,
                    metrics=self._create_failure_metrics(agent_task, cli_result.error_message or "CLI execution failed"),
                    overall_score=0.0,
                    passed=False,
                    timestamp=datetime.now(),
                    raw_output={}
                )
            
            # Extract agent-specific metrics from output
            extracted_metrics = self._extract_agent_metrics_from_output(cli_result.parsed_output or {})
            
            # Collect comprehensive agent metrics
            metrics = self._collect_agent_metrics_for_task(agent_task, extracted_metrics, start_time)
            
            # Calculate overall score
            overall_score = self._calculate_test_score(metrics)
            
            # Evaluate pass/fail based on agent success criteria
            passed = self._evaluate_agent_task_pass(metrics, agent_task.get('success_criteria', {}))
            
            return BenchmarkResult(
                test_id=agent_task['id'],
                category=agent_task.get('category', 'Unknown'),
                query=agent_task['query'],
                execution_time=time.time() - start_time,
                metrics=metrics,
                overall_score=overall_score,
                passed=passed,
                timestamp=datetime.now(),
                raw_output=cli_result.parsed_output or {}
            )
            
        except Exception as e:
            self.logger.error(f"Error running agent task {agent_task['id']}: {e}")
            return BenchmarkResult(
                test_id=agent_task['id'],
                category=agent_task.get('category', 'Unknown'),
                query=agent_task['query'],
                execution_time=time.time() - start_time,
                metrics=self._create_failure_metrics(agent_task, str(e)),
                overall_score=0.0,
                passed=False,
                timestamp=datetime.now(),
                raw_output={}
            )
    
    def _run_single_test_comprehensive(self, test_case: Dict[str, Any]) -> BenchmarkResult:
        """Run a single test case with comprehensive metric collection."""
        start_time = time.time()
        
        try:
            # Execute CLI command
            cli_result = self.cli_executor.execute_research(test_case['query'])
            
            if not cli_result.success:
                return BenchmarkResult(
                    test_id=test_case['id'],
                    category=test_case.get('category', 'Unknown'),
                    query=test_case['query'],
                    execution_time=time.time() - start_time,
                    metrics=self._create_failure_metrics(test_case, cli_result.error_message or "CLI execution failed"),
                    overall_score=0.0,
                    passed=False,
                    timestamp=datetime.now(),
                    raw_output={}
                )
            
            # Extract all metrics from output
            extracted_metrics = self._extract_all_metrics_from_output(cli_result.parsed_output or {})
            
            # Collect comprehensive metrics
            metrics = self._collect_metrics_for_test(test_case, extracted_metrics, start_time)
            
            # Calculate overall score
            overall_score = self._calculate_test_score(metrics)
            
            # Evaluate pass/fail
            passed = self._evaluate_test_pass(metrics, test_case.get('expected', {}))
            
            return BenchmarkResult(
                test_id=test_case['id'],
                category=test_case.get('category', 'Unknown'),
                query=test_case['query'],
                execution_time=time.time() - start_time,
                metrics=metrics,
                overall_score=overall_score,
                passed=passed,
                timestamp=datetime.now(),
                raw_output=cli_result.parsed_output or {}
            )
            
        except Exception as e:
            self.logger.error(f"Error running test {test_case['id']}: {e}")
            return BenchmarkResult(
                test_id=test_case['id'],
                category=test_case.get('category', 'Unknown'),
                query=test_case['query'],
                execution_time=time.time() - start_time,
                metrics=self._create_failure_metrics(test_case, str(e)),
                overall_score=0.0,
                passed=False,
                timestamp=datetime.now(),
                raw_output={}
            )
    
    def _extract_all_metrics_from_output(self, output: Dict[str, Any]) -> Dict[str, Any]:
        """Extract all possible metrics from CLI output - 실제 실행 결과만 수집."""
        extracted = {
            'sources': [],
            'execution_results': [],
            'creative_insights': [],
            'similar_research': [],
            'workflow_log': {}
        }
        
        # Extract sources (실제 소스만)
        if 'sources' in output and isinstance(output['sources'], list):
            extracted['sources'] = [
                source for source in output['sources'] 
                if not self._is_mock_data(source)
            ]
        
        # Extract execution results (실제 실행 결과만)
        if 'execution_results' in output and isinstance(output['execution_results'], list):
            extracted['execution_results'] = [
                result for result in output['execution_results']
                if not self._is_mock_data(result) and result.get('success', False)
            ]
        
        # Extract creative insights (실제 인사이트만)
        if 'creative_insights' in output and isinstance(output['creative_insights'], list):
            extracted['creative_insights'] = [
                insight for insight in output['creative_insights']
                if not self._is_mock_data(insight)
            ]
        
        # Extract similar research (실제 연구 결과만)
        if 'similar_research' in output and isinstance(output['similar_research'], list):
            extracted['similar_research'] = [
                research for research in output['similar_research']
                if not self._is_mock_data(research)
            ]
        
        # Extract workflow log (실제 워크플로우만)
        if 'workflow_log' in output and isinstance(output['workflow_log'], dict):
            extracted['workflow_log'] = output['workflow_log']
        
        return extracted
    
    def _is_mock_data(self, data: Any) -> bool:
        """Mock 데이터인지 확인."""
        if isinstance(data, dict):
            # Mock 데이터의 일반적인 패턴 확인
            mock_indicators = [
                'simulated', 'mock', 'dummy', 'fake', 'test_data',
                'placeholder', 'example', 'sample'
            ]
            
            # 키나 값에 mock 관련 문자열이 있는지 확인
            for key, value in data.items():
                if isinstance(key, str) and any(indicator in key.lower() for indicator in mock_indicators):
                    return True
                if isinstance(value, str) and any(indicator in value.lower() for indicator in mock_indicators):
                    return True
        
        elif isinstance(data, str):
            mock_indicators = ['simulated', 'mock', 'dummy', 'fake', 'test_data']
            return any(indicator in data.lower() for indicator in mock_indicators)
        
        return False
    
    def _extract_agent_metrics_from_output(self, output: Dict[str, Any]) -> Dict[str, Any]:
        """Extract agent-specific metrics from CLI output."""
        extracted = {
            'navigation_events': [],
            'tool_events': [],
            'agent_events': [],
            'reasoning_steps': [],
            'plan_steps': [],
            'execution_events': [],
            'reliability_events': []
        }
        
        # Get detailed_results from actual output structure
        detailed_results = output.get('detailed_results', {})
        
        # Extract navigation events (WebArena-style) - from execution_results
        execution_results = detailed_results.get('execution_results', [])
        if execution_results:
            # Convert execution results to navigation events
            for result in execution_results:
                if result.get('tool_used') == 'g-search' or result.get('tool_used') == 'web_search':
                    extracted['navigation_events'].append({
                        'status': 'success' if result.get('success', True) else 'failed',
                        'url': result.get('url', ''),
                        'content': result.get('content', '')
                    })
        
        # Extract tool usage events (ToolBench-style) - from execution_results
        for result in execution_results:
            if result.get('tool_used'):
                extracted['tool_events'].append({
                    'tool': result.get('tool_used'),
                    'status': 'success' if result.get('success', True) else 'failed',
                    'result': result.get('result', {})
                })
        
        # Extract agent collaboration events (AgentBench-style) - from agent_status or execution_plan
        agent_status = output.get('agent_status', {}) or detailed_results.get('agent_status', {})
        if agent_status and len(agent_status) > 1:
            extracted['agent_events'] = [
                {'agent_id': agent_id, 'status': status}
                for agent_id, status in agent_status.items()
            ]
        
        # Extract reasoning steps (ALFWorld-style) - from analyzed_objectives and planned_tasks
        analyzed_objectives = detailed_results.get('analyzed_objectives', [])
        planned_tasks = detailed_results.get('planned_tasks', [])
        
        if analyzed_objectives:
            extracted['reasoning_steps'] = [
                {'objective': obj.get('objective', ''), 'analysis': obj.get('analysis', {})}
                for obj in analyzed_objectives
            ]
        
        if planned_tasks:
            extracted['plan_steps'] = [
                {'task': task.get('task', ''), 'status': task.get('status', 'planned')}
                for task in planned_tasks
            ]
        
        # Extract execution events - use execution_results directly
        extracted['execution_events'] = execution_results
        
        # Extract reliability events - from quality_metrics
        quality_metrics = detailed_results.get('quality_metrics', {})
        if quality_metrics:
            extracted['reliability_events'] = [
                {'metric': metric, 'value': value}
                for metric, value in quality_metrics.items()
            ]
        
        # Fallback: Check for old format fields
        if 'navigation_log' in output:
            extracted['navigation_events'] = output['navigation_log']
        if 'tool_usage_log' in output:
            extracted['tool_events'] = output['tool_usage_log']
        if 'agent_collaboration_log' in output:
            extracted['agent_events'] = output['agent_collaboration_log']
        if 'reasoning_log' in output:
            extracted['reasoning_steps'] = output['reasoning_log']
        if 'planning_log' in output:
            extracted['plan_steps'] = output['planning_log']
        if 'execution_log' in output:
            extracted['execution_events'] = output['execution_log']
        if 'reliability_log' in output:
            extracted['reliability_events'] = output['reliability_log']
        
        return extracted
    
    def _collect_agent_metrics_for_task(self, agent_task: Dict[str, Any], 
                                       extracted_metrics: Dict[str, Any], 
                                       execution_time: float) -> List:
        """Collect ALL agent metrics for comprehensive evaluation."""
        metrics = []
        
        # Import agent metrics collector - try both absolute and relative imports
        try:
            from .agent_benchmark_metrics import AgentMetricsCollector
        except ImportError:
            try:
                from agent_benchmark_metrics import AgentMetricsCollector
            except ImportError:
                self.logger.error("Failed to import AgentMetricsCollector")
                return metrics
        
        agent_metrics_collector = AgentMetricsCollector(self.thresholds)
        
        # Import MetricResult for conversion - try both absolute and relative imports
        try:
            from .benchmark_metrics import MetricResult
        except ImportError:
            try:
                from benchmark_metrics import MetricResult
            except ImportError:
                self.logger.error("Failed to import MetricResult")
                return metrics
        
        # 1. Web Navigation Metrics (WebArena-style)
        if agent_task.get('category') == 'WebNavigation':
            navigation_events = extracted_metrics.get('navigation_events', [])
            nav_result = agent_metrics_collector.measure_navigation_success(navigation_events)
            metrics.append(MetricResult(
                name=nav_result.name,
                value=nav_result.value,
                threshold=nav_result.threshold,
                passed=nav_result.passed,
                category=nav_result.category,
                weight=nav_result.weight,
                metadata=nav_result.metadata
            ))
            
            # Information accuracy based on expected actions
            expected_actions = agent_task.get('expected_actions', [])
            info_result = agent_metrics_collector.measure_information_accuracy(
                navigation_events, expected_actions
            )
            metrics.append(MetricResult(
                name=info_result.name,
                value=info_result.value,
                threshold=info_result.threshold,
                passed=info_result.passed,
                category=info_result.category,
                weight=info_result.weight,
                metadata=info_result.metadata
            ))
        
        # 2. Tool Usage Metrics (ToolBench-style)
        if agent_task.get('category') == 'ToolUsage':
            tool_events = extracted_metrics.get('tool_events', [])
            tool_result = agent_metrics_collector.measure_tool_usage_success(tool_events)
            metrics.append(MetricResult(
                name=tool_result.name,
                value=tool_result.value,
                threshold=tool_result.threshold,
                passed=tool_result.passed,
                category=tool_result.category,
                weight=tool_result.weight,
                metadata=tool_result.metadata
            ))
            coord_result = agent_metrics_collector.measure_tool_coordination_efficiency(tool_events)
            metrics.append(MetricResult(
                name=coord_result.name,
                value=coord_result.value,
                threshold=coord_result.threshold,
                passed=coord_result.passed,
                category=coord_result.category,
                weight=coord_result.weight,
                metadata=coord_result.metadata
            ))
        
        # 3. Multi-Agent Collaboration Metrics (AgentBench-style)
        if agent_task.get('category') == 'MultiAgent':
            agent_events = extracted_metrics.get('agent_events', [])
            coord_result = agent_metrics_collector.measure_coordination_efficiency(agent_events)
            metrics.append(MetricResult(
                name=coord_result.name,
                value=coord_result.value,
                threshold=coord_result.threshold,
                passed=coord_result.passed,
                category=coord_result.category,
                weight=coord_result.weight,
                metadata=coord_result.metadata
            ))
            task_result = agent_metrics_collector.measure_task_completion_rate(agent_events)
            metrics.append(MetricResult(
                name=task_result.name,
                value=task_result.value,
                threshold=task_result.threshold,
                passed=task_result.passed,
                category=task_result.category,
                weight=task_result.weight,
                metadata=task_result.metadata
            ))
        
        # 4. Reasoning and Planning Metrics (ALFWorld-style)
        if agent_task.get('category') == 'Reasoning':
            reasoning_steps = extracted_metrics.get('reasoning_steps', [])
            reasoning_result = agent_metrics_collector.measure_reasoning_accuracy(reasoning_steps)
            metrics.append(MetricResult(
                name=reasoning_result.name,
                value=reasoning_result.value,
                threshold=reasoning_result.threshold,
                passed=reasoning_result.passed,
                category=reasoning_result.category,
                weight=reasoning_result.weight,
                metadata=reasoning_result.metadata
            ))
            
            plan_steps = extracted_metrics.get('plan_steps', [])
            plan_result = agent_metrics_collector.measure_plan_feasibility(plan_steps)
            metrics.append(MetricResult(
                name=plan_result.name,
                value=plan_result.value,
                threshold=plan_result.threshold,
                passed=plan_result.passed,
                category=plan_result.category,
                weight=plan_result.weight,
                metadata=plan_result.metadata
            ))
        
        # 5. Overall Agent Performance Metrics
        execution_events = extracted_metrics.get('execution_events', [])
        exec_result = agent_metrics_collector.measure_execution_efficiency(execution_events)
        metrics.append(MetricResult(
            name=exec_result.name,
            value=exec_result.value,
            threshold=exec_result.threshold,
            passed=exec_result.passed,
            category=exec_result.category,
            weight=exec_result.weight,
            metadata=exec_result.metadata
        ))
        
        reliability_events = extracted_metrics.get('reliability_events', [])
        reliability_result = agent_metrics_collector.measure_reliability_score(reliability_events)
        metrics.append(MetricResult(
            name=reliability_result.name,
            value=reliability_result.value,
            threshold=reliability_result.threshold,
            passed=reliability_result.passed,
            category=reliability_result.category,
            weight=reliability_result.weight,
            metadata=reliability_result.metadata
        ))
        
        return metrics
    
    def _evaluate_agent_task_pass(self, metrics: List, success_criteria: Dict[str, Any]) -> bool:
        """Evaluate if an agent task passes based on success criteria."""
        if not metrics:
            return False
        
        # Check for execution failures first
        execution_failures = [m for m in metrics if m.name == 'execution_failure']
        if execution_failures:
            return False
        
        # Check success criteria
        for criterion, expected_value in success_criteria.items():
            if criterion == 'min_sources':
                # Check if we have enough sources
                source_count = 0
                for metric in metrics:
                    if 'sources' in metric.metadata:
                        source_count = metric.metadata.get('total_sources', 0)
                        break
                if source_count < expected_value:
                    return False
            
            elif criterion == 'navigation_success_rate':
                nav_metric = next((m for m in metrics if m.name == 'navigation_success_rate'), None)
                if nav_metric and nav_metric.value < expected_value:
                    return False
            
            elif criterion == 'tool_usage_success_rate':
                tool_metric = next((m for m in metrics if m.name == 'tool_usage_success_rate'), None)
                if tool_metric and tool_metric.value < expected_value:
                    return False
            
            elif criterion == 'coordination_efficiency':
                coord_metric = next((m for m in metrics if m.name == 'coordination_efficiency'), None)
                if coord_metric and coord_metric.value < expected_value:
                    return False
            
            elif criterion == 'reasoning_accuracy':
                reasoning_metric = next((m for m in metrics if m.name == 'reasoning_accuracy'), None)
                if reasoning_metric and reasoning_metric.value < expected_value:
                    return False
        
        # Check if critical metrics pass their thresholds
        critical_metrics = ['execution_efficiency', 'reliability_score']
        for metric in metrics:
            if metric.name in critical_metrics and not metric.passed:
                return False
        
        return True
        
        self.logger.info(f"Completed {len(results)} benchmark tests")
        return results
    
    def _run_sequential_benchmarks(self, test_cases: List[Dict[str, Any]]) -> List[BenchmarkResult]:
        """Run benchmarks sequentially."""
        results = []
        
        for i, test_case in enumerate(test_cases, 1):
            self.logger.info(f"Running test {i}/{len(test_cases)}: {test_case['id']}")
            
            try:
                result = self._run_single_benchmark(test_case)
                results.append(result)
                
                if result.passed:
                    self.logger.info(f"✓ Test {test_case['id']} passed")
                else:
                    self.logger.warning(f"✗ Test {test_case['id']} failed")
                    
            except Exception as e:
                self.logger.error(f"Test {test_case['id']} failed with exception: {e}")
                # Create failed result
                failed_result = BenchmarkResult(
                    test_id=test_case['id'],
                    category=test_case.get('category', 'unknown'),
                    query=test_case['query'],
                    execution_time=0.0,
                    metrics=[],
                    overall_score=0.0,
                    passed=False,
                    timestamp=datetime.now(),
                    raw_output={'error': str(e)}
                )
                results.append(failed_result)
        
        return results
    
    def _run_parallel_benchmarks(self, test_cases: List[Dict[str, Any]], max_workers: int) -> List[BenchmarkResult]:
        """Run benchmarks in parallel."""
        results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_test = {
                executor.submit(self._run_single_benchmark, test_case): test_case
                for test_case in test_cases
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_test):
                test_case = future_to_test[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    if result.passed:
                        self.logger.info(f"✓ Test {test_case['id']} passed")
                    else:
                        self.logger.warning(f"✗ Test {test_case['id']} failed")
                        
                except Exception as e:
                    self.logger.error(f"Test {test_case['id']} failed with exception: {e}")
                    # Create failed result
                    failed_result = BenchmarkResult(
                        test_id=test_case['id'],
                        category=test_case.get('category', 'unknown'),
                        query=test_case['query'],
                        execution_time=0.0,
                        metrics=[],
                        overall_score=0.0,
                        passed=False,
                        timestamp=datetime.now(),
                        raw_output={'error': str(e)}
                    )
                    results.append(failed_result)
        
        return results
    
    def _run_single_benchmark(self, test_case: Dict[str, Any]) -> BenchmarkResult:
        """Run a single benchmark test case."""
        test_id = test_case['id']
        category = test_case.get('category', 'unknown')
        query = test_case['query']
        expected = test_case.get('expected', {})
        
        self.logger.info(f"Running benchmark: {test_id} - {query}")
        
        # Execute CLI command
        start_time = time.time()
        cli_result = self.cli_executor.execute_research(query)
        execution_time = time.time() - start_time
        
        # Extract metrics from output
        metrics = []
        if cli_result.success and cli_result.parsed_output:
            extracted_metrics = self.cli_executor.extract_metrics_from_output(cli_result.parsed_output)
            metrics = self._collect_metrics_for_test(test_case, extracted_metrics, execution_time)
        else:
            # Create failure metrics
            metrics = self._create_failure_metrics(test_case, cli_result.error_message)
        
        # Calculate overall score
        overall_score = self._calculate_test_score(metrics)
        
        # Determine if test passed
        passed = self._evaluate_test_pass(metrics, expected)
        
        # Clean up output files
        if cli_result.output_file:
            try:
                Path(cli_result.output_file).unlink()
            except Exception:
                pass
        
        return BenchmarkResult(
            test_id=test_id,
            category=category,
            query=query,
            execution_time=execution_time,
            metrics=metrics,
            overall_score=overall_score,
            passed=passed,
            timestamp=datetime.now(),
            raw_output=cli_result.parsed_output or {}
        )
    
    def _collect_metrics_for_test(self, test_case: Dict[str, Any], 
                                 extracted_metrics: Dict[str, Any], 
                                 execution_time: float) -> List:
        """Collect ALL metrics for comprehensive benchmark measurement."""
        metrics = []
        
        # 1. Performance Metrics
        metrics.append(self.metrics_collector.measure_response_time(
            time.time() - execution_time, time.time()
        ))
        
        # 2. Research Quality Metrics
        sources = extracted_metrics.get('sources', [])
        metrics.append(self.metrics_collector.measure_source_credibility(sources))
        
        execution_results = extracted_metrics.get('execution_results', [])
        claims = self._extract_claims_from_results(execution_results)
        metrics.append(self.metrics_collector.measure_factual_accuracy(claims, sources))
        
        # 3. Source Validation Metrics
        # Additional source validation metrics
        if sources:
            # Citation completeness (simplified)
            citation_metric = MetricResult(
                name="citation_completeness",
                value=min(1.0, len([s for s in sources if s.get('url')]) / max(1, len(sources))),
                threshold=self.thresholds.get('citation_completeness', 0.9),
                passed=len([s for s in sources if s.get('url')]) / max(1, len(sources)) >= self.thresholds.get('citation_completeness', 0.9),
                category="source_validation",
                metadata={"total_sources": len(sources), "sources_with_urls": len([s for s in sources if s.get('url')])}
            )
            metrics.append(citation_metric)
        
        # 4. Creative Insights Metrics
        creative_insights = extracted_metrics.get('creative_insights', [])
        metrics.extend(self.metrics_collector.measure_creative_quality(creative_insights))
        
        # 5. Memory & Learning Metrics
        similar_research = extracted_metrics.get('similar_research', [])
        if similar_research:
            metrics.append(self.metrics_collector.measure_memory_accuracy(similar_research, sources))
        
        # 6. Collaboration Metrics
        workflow_log = extracted_metrics.get('workflow_log', {})
        if workflow_log:
            metrics.extend(self.metrics_collector.measure_collaboration_efficiency(workflow_log))
        
        # 7. Additional Comprehensive Metrics
        # Information density
        if execution_results:
            total_content_length = sum(len(str(result.get('content', ''))) for result in execution_results)
            info_density = min(1.0, total_content_length / 10000)  # Normalize to 10k chars
            density_metric = MetricResult(
                name="information_density",
                value=info_density,
                threshold=self.thresholds.get('information_density', 0.7),
                passed=info_density >= self.thresholds.get('information_density', 0.7),
                category="research_quality",
                metadata={"total_content_length": total_content_length}
            )
            metrics.append(density_metric)
        
        # Analysis depth (based on number of insights and sources)
        analysis_depth = min(1.0, (len(creative_insights) * 0.2 + len(sources) * 0.1))
        depth_metric = MetricResult(
            name="analysis_depth",
            value=analysis_depth,
            threshold=self.thresholds.get('analysis_depth', 0.8),
            passed=analysis_depth >= self.thresholds.get('analysis_depth', 0.8),
            category="research_quality",
            metadata={"insights_count": len(creative_insights), "sources_count": len(sources)}
        )
        metrics.append(depth_metric)
        
        return metrics
    
    def _create_failure_metrics(self, test_case: Dict[str, Any], error_message: str) -> List:
        """Create failure metrics when CLI execution fails - 실제 실행 실패만 기록."""
        metrics = []
        
        # 실제 실행 실패만 기록 (Mock 데이터 제거)
        if "simulated" not in error_message.lower() and "mock" not in error_message.lower():
            metrics.append(MetricResult(
                name="execution_failure",
                value=0.0,
                threshold=1.0,
                passed=False,
                category="system",
                metadata={
                    "error": error_message, 
                    "test_id": test_case.get('id', 'unknown'),
                    "failure_type": "actual_execution_failure"
                }
            ))
            
            # 추가 실패 메트릭
            metrics.append(MetricResult(
                name="task_completion_rate",
                value=0.0,
                threshold=0.8,
                passed=False,
                category="performance",
                metadata={"test_id": test_case.get('id', 'unknown')}
            ))
            
            metrics.append(MetricResult(
                name="tool_execution_success",
                value=0.0,
                threshold=0.7,
                passed=False,
                category="reliability",
                metadata={"test_id": test_case.get('id', 'unknown')}
            ))
        
        return metrics
    
    def _extract_claims_from_results(self, execution_results: List[Dict[str, Any]]) -> List[str]:
        """Extract claims/statements from execution results."""
        claims = []
        
        for result in execution_results:
            if 'summary' in result:
                claims.append(result['summary'])
            if 'findings' in result:
                if isinstance(result['findings'], list):
                    claims.extend(result['findings'])
                else:
                    claims.append(str(result['findings']))
        
        return claims
    
    def _calculate_test_score(self, metrics: List) -> float:
        """Calculate overall score for a test based on metrics."""
        if not metrics:
            return 0.0
        
        # Group metrics by category
        category_scores = {}
        for metric in metrics:
            category = metric.category
            if category not in category_scores:
                category_scores[category] = []
            category_scores[category].append(metric.value)
        
        # Calculate average score per category
        avg_category_scores = {}
        for category, scores in category_scores.items():
            avg_category_scores[category] = sum(scores) / len(scores)
        
        # Calculate weighted overall score
        total_weight = 0.0
        weighted_sum = 0.0
        
        for category, avg_score in avg_category_scores.items():
            weight = self.weights.get(category, 1.0)
            weighted_sum += avg_score * weight
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def _evaluate_test_pass(self, metrics: List, expected: Dict[str, Any]) -> bool:
        """Evaluate if a test passes based on actual metrics and expected values."""
        if not metrics:
            return False
        
        # Check for execution failures first
        execution_failures = [m for m in metrics if m.name == 'execution_failure']
        if execution_failures:
            return False
        
        # Check if critical metrics pass their thresholds
        critical_metrics = ['response_time', 'source_credibility', 'factual_accuracy']
        critical_passed = True
        
        for metric in metrics:
            if metric.name in critical_metrics and not metric.passed:
                critical_passed = False
                break
        
        if not critical_passed:
            return False
        
        # Check expected values against actual results
        for key, expected_value in expected.items():
            if key == 'max_response_time':
                response_time_metric = next((m for m in metrics if m.name == 'response_time'), None)
                if response_time_metric and response_time_metric.value > expected_value:
                    return False
            elif key == 'min_credibility':
                credibility_metric = next((m for m in metrics if m.name == 'source_credibility'), None)
                if credibility_metric and credibility_metric.value < expected_value:
                    return False
            elif key == 'min_sources':
                # Check if we have enough sources from actual execution
                source_count = 0
                for metric in metrics:
                    if metric.name == 'source_credibility' and 'source_count' in metric.metadata:
                        source_count = metric.metadata['source_count']
                        break
                if source_count < expected_value:
                    return False
        
        return True
    
    def run_research_quality_benchmark(self) -> List[BenchmarkResult]:
        """Run research quality focused benchmarks."""
        test_cases = [tc for tc in self.config.get('test_cases', []) 
                     if tc.get('category') in ['Technology', 'Science', 'Health']]
        return self._run_sequential_benchmarks(test_cases)
    
    def run_performance_benchmark(self) -> List[BenchmarkResult]:
        """Run performance focused benchmarks."""
        test_cases = [tc for tc in self.config.get('test_cases', []) 
                     if tc.get('category') in ['Technology', 'Business']]
        return self._run_sequential_benchmarks(test_cases)
    
    def run_source_validation_benchmark(self) -> List[BenchmarkResult]:
        """Run source validation focused benchmarks."""
        test_cases = [tc for tc in self.config.get('test_cases', []) 
                     if tc.get('category') in ['Science', 'Health']]
        return self._run_sequential_benchmarks(test_cases)
    
    def run_creative_insights_benchmark(self) -> List[BenchmarkResult]:
        """Run creative insights focused benchmarks."""
        test_cases = [tc for tc in self.config.get('test_cases', []) 
                     if tc.get('category') in ['Creative', 'Health']]
        return self._run_sequential_benchmarks(test_cases)
    
    def run_memory_learning_benchmark(self) -> List[BenchmarkResult]:
        """Run memory and learning focused benchmarks."""
        test_cases = [tc for tc in self.config.get('test_cases', []) 
                     if tc.get('category') in ['Technology', 'Business']]
        return self._run_sequential_benchmarks(test_cases)
    
    def get_benchmark_summary(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Get summary statistics for benchmark results."""
        if not results:
            return {"error": "No results to summarize"}
        
        total_tests = len(results)
        passed_tests = sum(1 for r in results if r.passed)
        failed_tests = total_tests - passed_tests
        
        # Calculate average execution time
        avg_execution_time = sum(r.execution_time for r in results) / total_tests
        
        # Calculate overall score
        overall_score = self.analyzer.calculate_overall_score(results)
        
        # Get category breakdown
        category_stats = {}
        for result in results:
            category = result.category
            if category not in category_stats:
                category_stats[category] = {'total': 0, 'passed': 0, 'avg_score': 0.0}
            
            category_stats[category]['total'] += 1
            if result.passed:
                category_stats[category]['passed'] += 1
            category_stats[category]['avg_score'] += result.overall_score
        
        # Calculate averages
        for category in category_stats:
            stats = category_stats[category]
            stats['pass_rate'] = stats['passed'] / stats['total'] if stats['total'] > 0 else 0
            stats['avg_score'] = stats['avg_score'] / stats['total'] if stats['total'] > 0 else 0
        
        return {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "pass_rate": passed_tests / total_tests if total_tests > 0 else 0,
            "overall_score": overall_score,
            "average_execution_time": avg_execution_time,
            "category_breakdown": category_stats
        }
