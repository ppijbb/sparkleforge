"""
SparkleForge Benchmark Reporter

Generate comprehensive reports from benchmark results including JSON, Markdown,
and console output formats for CI/CD integration and documentation.
"""

import json
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime
import statistics

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    plt = None
    sns = None
    pd = None

try:
    from .benchmark_metrics import BenchmarkResult, BenchmarkAnalyzer
except ImportError:
    from benchmark_metrics import BenchmarkResult, BenchmarkAnalyzer

logger = logging.getLogger(__name__)


class BenchmarkReporter:
    """Generate comprehensive benchmark reports."""
    
    def __init__(self, output_dir: str = "results"):
        # Convert to absolute path to avoid issues
        output_path = Path(output_dir)
        if not output_path.is_absolute():
            # Get project root (assuming we're in tests/benchmark/)
            current_dir = Path(__file__).parent
            project_root = current_dir.parent.parent
            output_path = project_root / output_dir
        
        self.output_dir = output_path
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        # Create charts subdirectory
        self.charts_dir = self.output_dir / "charts"
        self.charts_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_json_report(self, results: List[BenchmarkResult], 
                           metadata: Optional[Dict[str, Any]] = None) -> str:
        """Generate JSON report for CI/CD integration."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"benchmark_results_{timestamp}.json"
        filepath = self.output_dir / filename
        
        # Prepare report data
        report_data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_tests": len(results),
                "version": "2.1.0",
                "generator": "SparkleForge Benchmark Reporter"
            },
            "summary": self._generate_summary_stats(results),
            "results": [self._serialize_result(result) for result in results],
            "categories": self._analyze_by_category(results),
            "performance": self._analyze_performance(results)
        }
        
        # Add custom metadata if provided
        if metadata:
            report_data["metadata"].update(metadata)
        
        # Write JSON file
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, default=str)
            
            self.logger.info(f"JSON report generated: {filepath}")
            return str(filepath)
        except Exception as e:
            self.logger.error(f"Failed to write JSON report to {filepath}: {e}")
            # Try alternative location
            alt_filepath = Path.cwd() / filename
            try:
                with open(alt_filepath, 'w', encoding='utf-8') as f:
                    json.dump(report_data, f, indent=2, default=str)
                self.logger.info(f"JSON report generated at alternative location: {alt_filepath}")
                return str(alt_filepath)
            except Exception as e2:
                self.logger.error(f"Failed to write JSON report to alternative location {alt_filepath}: {e2}")
                raise
    
    def generate_markdown_report(self, results: List[BenchmarkResult], 
                               metadata: Optional[Dict[str, Any]] = None) -> str:
        """Generate Markdown report for documentation."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"benchmark_report_{timestamp}.md"
        filepath = self.output_dir / filename
        
        # Generate charts first
        chart_paths = self._generate_charts(results)
        
        # Generate markdown content
        markdown_content = self._build_markdown_content(results, chart_paths, metadata)
        
        # Write markdown file
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            
            self.logger.info(f"Markdown report generated: {filepath}")
            return str(filepath)
        except Exception as e:
            self.logger.error(f"Failed to write Markdown report to {filepath}: {e}")
            # Try alternative location
            alt_filepath = Path.cwd() / filename
            try:
                with open(alt_filepath, 'w', encoding='utf-8') as f:
                    f.write(markdown_content)
                self.logger.info(f"Markdown report generated at alternative location: {alt_filepath}")
                return str(alt_filepath)
            except Exception as e2:
                self.logger.error(f"Failed to write Markdown report to alternative location {alt_filepath}: {e2}")
                raise
    
    def generate_console_summary(self, results: List[BenchmarkResult]) -> str:
        """Generate console summary for quick review."""
        if not results:
            return "No benchmark results to summarize."
        
        summary = self._generate_summary_stats(results)
        
        # Build console output
        output = []
        output.append("=" * 60)
        output.append("SPARKLEFORGE BENCHMARK SUMMARY")
        output.append("=" * 60)
        output.append("")
        
        # Overall statistics
        output.append(f"Total Tests: {summary['total_tests']}")
        output.append(f"Passed: {summary['passed_tests']} ({summary['pass_rate']:.1%})")
        output.append(f"Failed: {summary['failed_tests']} ({1-summary['pass_rate']:.1%})")
        output.append(f"Overall Score: {summary['overall_score']:.3f}")
        output.append(f"Average Execution Time: {summary['average_execution_time']:.1f}s")
        output.append("")
        
        # Category breakdown
        output.append("CATEGORY BREAKDOWN:")
        output.append("-" * 30)
        for category, stats in summary['category_breakdown'].items():
            output.append(f"{category:15} | {stats['passed']:2}/{stats['total']:2} | {stats['pass_rate']:.1%} | {stats['avg_score']:.3f}")
        output.append("")
        
        # Performance insights
        if summary['average_execution_time'] > 120:
            output.append("⚠️  WARNING: Average execution time is high")
        if summary['pass_rate'] < 0.8:
            output.append("⚠️  WARNING: Pass rate is below 80%")
        if summary['overall_score'] < 0.7:
            output.append("⚠️  WARNING: Overall score is below 70%")
        
        # Failed tests
        failed_tests = [r for r in results if not r.passed]
        if failed_tests:
            output.append("")
            output.append("FAILED TESTS:")
            output.append("-" * 20)
            for result in failed_tests:
                output.append(f"  {result.test_id} ({result.category}) - Score: {result.overall_score:.3f}")
        
        output.append("")
        output.append("=" * 60)
        
        return "\n".join(output)
    
    def create_comparison_chart(self, current_results: List[BenchmarkResult], 
                              threshold_results: Optional[List[BenchmarkResult]] = None) -> str:
        """Create comparison chart between current and threshold results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"comparison_chart_{timestamp}.png"
        filepath = self.charts_dir / filename
        
        # Prepare data for comparison
        current_summary = self._generate_summary_stats(current_results)
        
        if threshold_results:
            threshold_summary = self._generate_summary_stats(threshold_results)
            self._create_comparison_radar_chart(current_summary, threshold_summary, filepath)
        else:
            self._create_performance_radar_chart(current_summary, filepath)
        
        self.logger.info(f"Comparison chart generated: {filepath}")
        return str(filepath)
    
    def _generate_summary_stats(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Generate summary statistics from results."""
        if not results:
            return {
                "total_tests": 0,
                "passed_tests": 0,
                "failed_tests": 0,
                "pass_rate": 0.0,
                "overall_score": 0.0,
                "average_execution_time": 0.0,
                "category_breakdown": {}
            }
        
        total_tests = len(results)
        passed_tests = sum(1 for r in results if r.passed)
        failed_tests = total_tests - passed_tests
        pass_rate = passed_tests / total_tests if total_tests > 0 else 0.0
        
        # Calculate overall score
        overall_score = statistics.mean([r.overall_score for r in results]) if results else 0.0
        
        # Calculate average execution time
        avg_execution_time = statistics.mean([r.execution_time for r in results]) if results else 0.0
        
        # Category breakdown
        category_breakdown = {}
        for result in results:
            category = result.category
            if category not in category_breakdown:
                category_breakdown[category] = {
                    "total": 0,
                    "passed": 0,
                    "avg_score": 0.0,
                    "avg_execution_time": 0.0
                }
            
            category_breakdown[category]["total"] += 1
            if result.passed:
                category_breakdown[category]["passed"] += 1
            category_breakdown[category]["avg_score"] += result.overall_score
            category_breakdown[category]["avg_execution_time"] += result.execution_time
        
        # Calculate averages
        for category in category_breakdown:
            stats = category_breakdown[category]
            stats["pass_rate"] = stats["passed"] / stats["total"] if stats["total"] > 0 else 0.0
            stats["avg_score"] = stats["avg_score"] / stats["total"] if stats["total"] > 0 else 0.0
            stats["avg_execution_time"] = stats["avg_execution_time"] / stats["total"] if stats["total"] > 0 else 0.0
        
        return {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "pass_rate": pass_rate,
            "overall_score": overall_score,
            "average_execution_time": avg_execution_time,
            "category_breakdown": category_breakdown
        }
    
    def _serialize_result(self, result: BenchmarkResult) -> Dict[str, Any]:
        """Serialize BenchmarkResult to dictionary."""
        return {
            "test_id": result.test_id,
            "category": result.category,
            "query": result.query,
            "execution_time": result.execution_time,
            "overall_score": result.overall_score,
            "passed": result.passed,
            "timestamp": result.timestamp.isoformat() if result.timestamp else None,
            "metrics": [
                {
                    "name": m.name,
                    "value": m.value,
                    "threshold": m.threshold,
                    "passed": m.passed,
                    "category": m.category,
                    "metadata": m.metadata
                } for m in result.metrics
            ]
        }
    
    def _analyze_by_category(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Analyze results by category."""
        category_analysis = {}
        
        for result in results:
            category = result.category
            if category not in category_analysis:
                category_analysis[category] = {
                    "total_tests": 0,
                    "passed_tests": 0,
                    "avg_score": 0.0,
                    "avg_execution_time": 0.0,
                    "metrics_breakdown": {}
                }
            
            stats = category_analysis[category]
            stats["total_tests"] += 1
            if result.passed:
                stats["passed_tests"] += 1
            stats["avg_score"] += result.overall_score
            stats["avg_execution_time"] += result.execution_time
            
            # Analyze metrics
            for metric in result.metrics:
                metric_name = metric.name
                if metric_name not in stats["metrics_breakdown"]:
                    stats["metrics_breakdown"][metric_name] = {
                        "total": 0,
                        "passed": 0,
                        "avg_value": 0.0
                    }
                
                metric_stats = stats["metrics_breakdown"][metric_name]
                metric_stats["total"] += 1
                if metric.passed:
                    metric_stats["passed"] += 1
                metric_stats["avg_value"] += metric.value
        
        # Calculate averages
        for category in category_analysis:
            stats = category_analysis[category]
            stats["pass_rate"] = stats["passed_tests"] / stats["total_tests"] if stats["total_tests"] > 0 else 0.0
            stats["avg_score"] = stats["avg_score"] / stats["total_tests"] if stats["total_tests"] > 0 else 0.0
            stats["avg_execution_time"] = stats["avg_execution_time"] / stats["total_tests"] if stats["total_tests"] > 0 else 0.0
            
            # Calculate metric averages
            for metric_name, metric_stats in stats["metrics_breakdown"].items():
                metric_stats["pass_rate"] = metric_stats["passed"] / metric_stats["total"] if metric_stats["total"] > 0 else 0.0
                metric_stats["avg_value"] = metric_stats["avg_value"] / metric_stats["total"] if metric_stats["total"] > 0 else 0.0
        
        return category_analysis
    
    def _analyze_performance(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Analyze performance characteristics."""
        if not results:
            return {}
        
        execution_times = [r.execution_time for r in results]
        scores = [r.overall_score for r in results]
        
        return {
            "execution_time": {
                "min": min(execution_times),
                "max": max(execution_times),
                "mean": statistics.mean(execution_times),
                "median": statistics.median(execution_times),
                "std": statistics.stdev(execution_times) if len(execution_times) > 1 else 0.0
            },
            "scores": {
                "min": min(scores),
                "max": max(scores),
                "mean": statistics.mean(scores),
                "median": statistics.median(scores),
                "std": statistics.stdev(scores) if len(scores) > 1 else 0.0
            }
        }
    
    def _generate_charts(self, results: List[BenchmarkResult]) -> Dict[str, str]:
        """Generate visualization charts."""
        chart_paths = {}
        
        if not results:
            return chart_paths
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Response time distribution
        chart_paths["response_time"] = self._create_response_time_chart(results, timestamp)
        
        # Quality metrics radar chart
        chart_paths["quality_metrics"] = self._create_quality_radar_chart(results, timestamp)
        
        # Source credibility histogram
        chart_paths["source_credibility"] = self._create_credibility_histogram(results, timestamp)
        
        # Category performance comparison
        chart_paths["category_performance"] = self._create_category_comparison_chart(results, timestamp)
        
        return chart_paths
    
    def _create_response_time_chart(self, results: List[BenchmarkResult], timestamp: str) -> str:
        """Create response time distribution chart."""
        if not PLOTTING_AVAILABLE:
            return ""
            
        filename = f"response_time_distribution_{timestamp}.png"
        filepath = self.charts_dir / filename
        
        execution_times = [r.execution_time for r in results]
        
        plt.figure(figsize=(10, 6))
        plt.hist(execution_times, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(statistics.mean(execution_times), color='red', linestyle='--', 
                   label=f'Mean: {statistics.mean(execution_times):.1f}s')
        plt.xlabel('Execution Time (seconds)')
        plt.ylabel('Frequency')
        plt.title('Response Time Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(filepath)
    
    def _create_quality_radar_chart(self, results: List[BenchmarkResult], timestamp: str) -> str:
        """Create quality metrics radar chart."""
        if not PLOTTING_AVAILABLE:
            return ""
            
        filename = f"quality_metrics_radar_{timestamp}.png"
        filepath = self.charts_dir / filename
        
        # Extract metrics by category
        category_metrics = {}
        for result in results:
            for metric in result.metrics:
                category = metric.category
                if category not in category_metrics:
                    category_metrics[category] = []
                category_metrics[category].append(metric.value)
        
        # Calculate average scores per category
        avg_scores = {}
        for category, values in category_metrics.items():
            avg_scores[category] = statistics.mean(values) if values else 0.0
        
        # Create radar chart
        categories = list(avg_scores.keys())
        scores = list(avg_scores.values())
        
        # Add first category to end to close the radar
        categories += [categories[0]]
        scores += [scores[0]]
        
        angles = [n / len(categories) * 2 * 3.14159 for n in range(len(categories))]
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        ax.plot(angles, scores, 'o-', linewidth=2, label='Average Score')
        ax.fill(angles, scores, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories[:-1])
        ax.set_ylim(0, 1)
        ax.set_title('Quality Metrics Radar Chart', size=16, pad=20)
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(filepath)
    
    def _create_credibility_histogram(self, results: List[BenchmarkResult], timestamp: str) -> str:
        """Create source credibility histogram."""
        if not PLOTTING_AVAILABLE:
            return ""
            
        filename = f"source_credibility_histogram_{timestamp}.png"
        filepath = self.charts_dir / filename
        
        # Extract credibility scores
        credibility_scores = []
        for result in results:
            for metric in result.metrics:
                if metric.name == "source_credibility":
                    credibility_scores.append(metric.value)
        
        if not credibility_scores:
            return ""
        
        plt.figure(figsize=(10, 6))
        plt.hist(credibility_scores, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
        plt.axvline(statistics.mean(credibility_scores), color='red', linestyle='--',
                   label=f'Mean: {statistics.mean(credibility_scores):.3f}')
        plt.axvline(0.7, color='orange', linestyle=':', label='Threshold: 0.7')
        plt.xlabel('Source Credibility Score')
        plt.ylabel('Frequency')
        plt.title('Source Credibility Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(filepath)
    
    def _create_category_comparison_chart(self, results: List[BenchmarkResult], timestamp: str) -> str:
        """Create category performance comparison chart."""
        if not PLOTTING_AVAILABLE:
            return ""
            
        filename = f"category_performance_{timestamp}.png"
        filepath = self.charts_dir / filename
        
        # Group results by category
        category_data = {}
        for result in results:
            category = result.category
            if category not in category_data:
                category_data[category] = {"scores": [], "times": []}
            category_data[category]["scores"].append(result.overall_score)
            category_data[category]["times"].append(result.execution_time)
        
        # Prepare data for plotting
        categories = list(category_data.keys())
        avg_scores = [statistics.mean(category_data[cat]["scores"]) for cat in categories]
        avg_times = [statistics.mean(category_data[cat]["times"]) for cat in categories]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Score comparison
        ax1.bar(categories, avg_scores, color='skyblue', alpha=0.7)
        ax1.set_ylabel('Average Score')
        ax1.set_title('Average Score by Category')
        ax1.set_ylim(0, 1)
        ax1.tick_params(axis='x', rotation=45)
        
        # Time comparison
        ax2.bar(categories, avg_times, color='lightcoral', alpha=0.7)
        ax2.set_ylabel('Average Execution Time (s)')
        ax2.set_title('Average Execution Time by Category')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(filepath)
    
    def _create_comparison_radar_chart(self, current_summary: Dict, threshold_summary: Dict, filepath: Path) -> None:
        """Create comparison radar chart between current and threshold results."""
        if not PLOTTING_AVAILABLE:
            return
            
        categories = list(current_summary['category_breakdown'].keys())
        current_scores = [current_summary['category_breakdown'][cat]['avg_score'] for cat in categories]
        threshold_scores = [threshold_summary['category_breakdown'].get(cat, {}).get('avg_score', 0) for cat in categories]
        
        # Add first category to end to close the radar
        categories += [categories[0]]
        current_scores += [current_scores[0]]
        threshold_scores += [threshold_scores[0]]
        
        angles = [n / len(categories) * 2 * 3.14159 for n in range(len(categories))]
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        ax.plot(angles, current_scores, 'o-', linewidth=2, label='Current', color='blue')
        ax.plot(angles, threshold_scores, 'o-', linewidth=2, label='Threshold', color='red')
        ax.fill(angles, current_scores, alpha=0.25, color='blue')
        ax.fill(angles, threshold_scores, alpha=0.25, color='red')
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories[:-1])
        ax.set_ylim(0, 1)
        ax.set_title('Current vs Threshold Performance', size=16, pad=20)
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_performance_radar_chart(self, summary: Dict, filepath: Path) -> None:
        """Create performance radar chart."""
        if not PLOTTING_AVAILABLE:
            return
            
        categories = list(summary['category_breakdown'].keys())
        scores = [summary['category_breakdown'][cat]['avg_score'] for cat in categories]
        
        # Add first category to end to close the radar
        categories += [categories[0]]
        scores += [scores[0]]
        
        angles = [n / len(categories) * 2 * 3.14159 for n in range(len(categories))]
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        ax.plot(angles, scores, 'o-', linewidth=2, color='green')
        ax.fill(angles, scores, alpha=0.25, color='green')
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories[:-1])
        ax.set_ylim(0, 1)
        ax.set_title('Performance Overview', size=16, pad=20)
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _build_markdown_content(self, results: List[BenchmarkResult], 
                              chart_paths: Dict[str, str], 
                              metadata: Optional[Dict[str, Any]]) -> str:
        """Build markdown report content."""
        summary = self._generate_summary_stats(results)
        category_analysis = self._analyze_by_category(results)
        performance_analysis = self._analyze_performance(results)
        
        content = []
        content.append("# SparkleForge Benchmark Report")
        content.append("")
        content.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        content.append(f"**Version:** 2.1.0")
        content.append("")
        
        # Executive Summary
        content.append("## Executive Summary")
        content.append("")
        content.append(f"- **Total Tests:** {summary['total_tests']}")
        content.append(f"- **Pass Rate:** {summary['pass_rate']:.1%}")
        content.append(f"- **Overall Score:** {summary['overall_score']:.3f}")
        content.append(f"- **Average Execution Time:** {summary['average_execution_time']:.1f}s")
        content.append("")
        
        # Performance Overview
        content.append("## Performance Overview")
        content.append("")
        if chart_paths.get("quality_metrics"):
            content.append(f"![Quality Metrics]({chart_paths['quality_metrics']})")
            content.append("")
        
        # Category Analysis
        content.append("## Category Analysis")
        content.append("")
        content.append("| Category | Tests | Passed | Pass Rate | Avg Score | Avg Time |")
        content.append("|----------|-------|--------|-----------|-----------|----------|")
        
        for category, stats in category_analysis.items():
            content.append(f"| {category} | {stats['total_tests']} | {stats['passed_tests']} | "
                          f"{stats['pass_rate']:.1%} | {stats['avg_score']:.3f} | "
                          f"{stats['avg_execution_time']:.1f}s |")
        content.append("")
        
        # Detailed Results
        content.append("## Detailed Results")
        content.append("")
        for result in results:
            status = "✅ PASS" if result.passed else "❌ FAIL"
            content.append(f"### {result.test_id} - {status}")
            content.append(f"- **Category:** {result.category}")
            content.append(f"- **Query:** {result.query}")
            content.append(f"- **Score:** {result.overall_score:.3f}")
            content.append(f"- **Execution Time:** {result.execution_time:.1f}s")
            content.append("")
            
            if result.metrics:
                content.append("**Metrics:**")
                content.append("")
                content.append("| Metric | Value | Threshold | Status |")
                content.append("|--------|-------|-----------|--------|")
                
                for metric in result.metrics:
                    status_icon = "✅" if metric.passed else "❌"
                    content.append(f"| {metric.name} | {metric.value:.3f} | {metric.threshold:.3f} | {status_icon} |")
                content.append("")
        
        # Charts
        if chart_paths:
            content.append("## Visualizations")
            content.append("")
            
            if chart_paths.get("response_time"):
                content.append("### Response Time Distribution")
                content.append(f"![Response Time]({chart_paths['response_time']})")
                content.append("")
            
            if chart_paths.get("source_credibility"):
                content.append("### Source Credibility Distribution")
                content.append(f"![Source Credibility]({chart_paths['source_credibility']})")
                content.append("")
            
            if chart_paths.get("category_performance"):
                content.append("### Category Performance Comparison")
                content.append(f"![Category Performance]({chart_paths['category_performance']})")
                content.append("")
        
        # Recommendations
        content.append("## Recommendations")
        content.append("")
        
        if summary['pass_rate'] < 0.8:
            content.append("- ⚠️ **Low Pass Rate:** Focus on improving test reliability")
        
        if summary['average_execution_time'] > 120:
            content.append("- ⚠️ **High Execution Time:** Optimize performance bottlenecks")
        
        if summary['overall_score'] < 0.7:
            content.append("- ⚠️ **Low Overall Score:** Review and improve core functionality")
        
        # Find worst performing category
        worst_category = min(category_analysis.items(), 
                           key=lambda x: x[1]['avg_score']) if category_analysis else None
        if worst_category and worst_category[1]['avg_score'] < 0.6:
            content.append(f"- 🔍 **Focus Area:** {worst_category[0]} category needs attention")
        
        content.append("")
        content.append("---")
        content.append("*Generated by SparkleForge Benchmark System*")
        
        return "\n".join(content)
