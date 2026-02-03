"""
SparkleForge Benchmark Comparator

Compare benchmark results against baselines and detect regressions.
Provides historical analysis and improvement tracking.
"""

import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
import statistics
from dataclasses import dataclass

try:
    from .benchmark_metrics import BenchmarkResult, BenchmarkAnalyzer
except ImportError:
    from benchmark_metrics import BenchmarkResult, BenchmarkAnalyzer

logger = logging.getLogger(__name__)


@dataclass
class Regression:
    """Represents a performance regression."""
    test_id: str
    type: str  # 'performance_degradation', 'quality_degradation', 'new_failure'
    current_value: float
    previous_value: float
    degradation_percent: float
    severity: str  # 'low', 'medium', 'high', 'critical'
    description: str


@dataclass
class ComparisonResult:
    """Result of comparing current results with baseline."""
    current_summary: Dict[str, Any]
    baseline_summary: Dict[str, Any]
    regressions: List[Regression]
    improvements: List[Dict[str, Any]]
    overall_change: float
    status: str  # 'improved', 'regressed', 'stable'


class BenchmarkComparator:
    """Compare benchmark results against baselines and detect regressions."""
    
    def __init__(self, baselines_dir: str = "tests/baselines"):
        self.baselines_dir = Path(baselines_dir)
        self.baselines_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)
    
    def load_historical_results(self, version: Optional[str] = None) -> List[BenchmarkResult]:
        """Load historical benchmark results."""
        if version:
            baseline_file = self.baselines_dir / f"baseline_{version}.json"
        else:
            # Load the most recent baseline
            baseline_files = list(self.baselines_dir.glob("baseline_*.json"))
            if not baseline_files:
                return []
            baseline_file = max(baseline_files, key=lambda f: f.stat().st_mtime)
        
        if not baseline_file.exists():
            self.logger.warning(f"Baseline file not found: {baseline_file}")
            return []
        
        try:
            with open(baseline_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Convert back to BenchmarkResult objects
            results = []
            for result_data in data.get('results', []):
                result = BenchmarkResult(
                    test_id=result_data['test_id'],
                    category=result_data['category'],
                    query=result_data['query'],
                    execution_time=result_data['execution_time'],
                    metrics=[],  # Simplified for comparison
                    overall_score=result_data['overall_score'],
                    passed=result_data['passed'],
                    timestamp=datetime.fromisoformat(result_data['timestamp']) if result_data.get('timestamp') else None
                )
                results.append(result)
            
            self.logger.info(f"Loaded {len(results)} historical results from {baseline_file}")
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to load historical results: {e}")
            return []
    
    def save_baseline(self, results: List[BenchmarkResult], version: str) -> str:
        """Save current results as a new baseline."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        baseline_file = self.baselines_dir / f"baseline_{version}_{timestamp}.json"
        
        # Convert results to serializable format
        baseline_data = {
            "version": version,
            "timestamp": datetime.now().isoformat(),
            "total_tests": len(results),
            "results": [
                {
                    "test_id": r.test_id,
                    "category": r.category,
                    "query": r.query,
                    "execution_time": r.execution_time,
                    "overall_score": r.overall_score,
                    "passed": r.passed,
                    "timestamp": r.timestamp.isoformat() if r.timestamp else None
                }
                for r in results
            ]
        }
        
        try:
            with open(baseline_file, 'w', encoding='utf-8') as f:
                json.dump(baseline_data, f, indent=2, default=str)
            
            self.logger.info(f"Saved baseline: {baseline_file}")
            return str(baseline_file)
            
        except Exception as e:
            self.logger.error(f"Failed to save baseline: {e}")
            return ""
    
    def compare_with_threshold(self, current_results: List[BenchmarkResult], 
                             thresholds: Dict[str, float]) -> ComparisonResult:
        """Compare current results against fixed thresholds."""
        current_summary = self._generate_summary_stats(current_results)
        
        # Create a mock baseline with threshold values
        baseline_summary = {
            "overall_score": thresholds.get('overall_score', 0.7),
            "pass_rate": thresholds.get('pass_rate', 0.8),
            "average_execution_time": thresholds.get('max_execution_time', 120.0)
        }
        
        regressions = []
        improvements = []
        
        # Check overall score
        if current_summary['overall_score'] < baseline_summary['overall_score']:
            regressions.append(Regression(
                test_id="overall",
                type="quality_degradation",
                current_value=current_summary['overall_score'],
                previous_value=baseline_summary['overall_score'],
                degradation_percent=((baseline_summary['overall_score'] - current_summary['overall_score']) / baseline_summary['overall_score']) * 100,
                severity=self._calculate_severity(current_summary['overall_score'], baseline_summary['overall_score']),
                description="Overall score below threshold"
            ))
        elif current_summary['overall_score'] > baseline_summary['overall_score']:
            improvements.append({
                "metric": "overall_score",
                "current": current_summary['overall_score'],
                "threshold": baseline_summary['overall_score'],
                "improvement_percent": ((current_summary['overall_score'] - baseline_summary['overall_score']) / baseline_summary['overall_score']) * 100
            })
        
        # Check pass rate
        if current_summary['pass_rate'] < baseline_summary['pass_rate']:
            regressions.append(Regression(
                test_id="overall",
                type="quality_degradation",
                current_value=current_summary['pass_rate'],
                previous_value=baseline_summary['pass_rate'],
                degradation_percent=((baseline_summary['pass_rate'] - current_summary['pass_rate']) / baseline_summary['pass_rate']) * 100,
                severity=self._calculate_severity(current_summary['pass_rate'], baseline_summary['pass_rate']),
                description="Pass rate below threshold"
            ))
        elif current_summary['pass_rate'] > baseline_summary['pass_rate']:
            improvements.append({
                "metric": "pass_rate",
                "current": current_summary['pass_rate'],
                "threshold": baseline_summary['pass_rate'],
                "improvement_percent": ((current_summary['pass_rate'] - baseline_summary['pass_rate']) / baseline_summary['pass_rate']) * 100
            })
        
        # Check execution time
        if current_summary['average_execution_time'] > baseline_summary['average_execution_time']:
            regressions.append(Regression(
                test_id="overall",
                type="performance_degradation",
                current_value=current_summary['average_execution_time'],
                previous_value=baseline_summary['average_execution_time'],
                degradation_percent=((current_summary['average_execution_time'] - baseline_summary['average_execution_time']) / baseline_summary['average_execution_time']) * 100,
                severity=self._calculate_severity(baseline_summary['average_execution_time'], current_summary['average_execution_time']),
                description="Execution time above threshold"
            ))
        elif current_summary['average_execution_time'] < baseline_summary['average_execution_time']:
            improvements.append({
                "metric": "execution_time",
                "current": current_summary['average_execution_time'],
                "threshold": baseline_summary['average_execution_time'],
                "improvement_percent": ((baseline_summary['average_execution_time'] - current_summary['average_execution_time']) / baseline_summary['average_execution_time']) * 100
            })
        
        # Determine overall status
        if regressions:
            status = "regressed"
        elif improvements:
            status = "improved"
        else:
            status = "stable"
        
        overall_change = current_summary['overall_score'] - baseline_summary['overall_score']
        
        return ComparisonResult(
            current_summary=current_summary,
            baseline_summary=baseline_summary,
            regressions=regressions,
            improvements=improvements,
            overall_change=overall_change,
            status=status
        )
    
    def detect_regression(self, current_results: List[BenchmarkResult], 
                         previous_results: List[BenchmarkResult]) -> List[Regression]:
        """Detect regressions compared to previous results."""
        if not previous_results:
            return []
        
        regressions = []
        
        # Create lookup for previous results
        prev_lookup = {r.test_id: r for r in previous_results}
        
        for current in current_results:
            if current.test_id not in prev_lookup:
                # New test - not a regression
                continue
            
            prev = prev_lookup[current.test_id]
            
            # Check for performance degradation
            if current.execution_time > prev.execution_time * 1.1:  # 10% slower
                regressions.append(Regression(
                    test_id=current.test_id,
                    type="performance_degradation",
                    current_value=current.execution_time,
                    previous_value=prev.execution_time,
                    degradation_percent=((current.execution_time - prev.execution_time) / prev.execution_time) * 100,
                    severity=self._calculate_severity(prev.execution_time, current.execution_time),
                    description=f"Execution time increased from {prev.execution_time:.1f}s to {current.execution_time:.1f}s"
                ))
            
            # Check for quality degradation
            if current.overall_score < prev.overall_score * 0.95:  # 5% lower quality
                regressions.append(Regression(
                    test_id=current.test_id,
                    type="quality_degradation",
                    current_value=current.overall_score,
                    previous_value=prev.overall_score,
                    degradation_percent=((prev.overall_score - current.overall_score) / prev.overall_score) * 100,
                    severity=self._calculate_severity(current.overall_score, prev.overall_score),
                    description=f"Overall score decreased from {prev.overall_score:.3f} to {current.overall_score:.3f}"
                ))
            
            # Check for new failures
            if prev.passed and not current.passed:
                regressions.append(Regression(
                    test_id=current.test_id,
                    type="new_failure",
                    current_value=0.0,
                    previous_value=1.0,
                    degradation_percent=100.0,
                    severity="critical",
                    description="Test that previously passed is now failing"
                ))
        
        return regressions
    
    def generate_improvement_report(self, current_results: List[BenchmarkResult], 
                                  previous_results: List[BenchmarkResult]) -> str:
        """Generate improvement report comparing current and previous results."""
        if not previous_results:
            return "No previous results available for comparison."
        
        current_summary = self._generate_summary_stats(current_results)
        previous_summary = self._generate_summary_stats(previous_results)
        
        regressions = self.detect_regression(current_results, previous_results)
        
        # Calculate improvements
        improvements = []
        
        # Overall score improvement
        if current_summary['overall_score'] > previous_summary['overall_score']:
            improvement_percent = ((current_summary['overall_score'] - previous_summary['overall_score']) / previous_summary['overall_score']) * 100
            improvements.append(f"Overall score improved by {improvement_percent:.1f}%")
        
        # Pass rate improvement
        if current_summary['pass_rate'] > previous_summary['pass_rate']:
            improvement_percent = ((current_summary['pass_rate'] - previous_summary['pass_rate']) / previous_summary['pass_rate']) * 100
            improvements.append(f"Pass rate improved by {improvement_percent:.1f}%")
        
        # Execution time improvement
        if current_summary['average_execution_time'] < previous_summary['average_execution_time']:
            improvement_percent = ((previous_summary['average_execution_time'] - current_summary['average_execution_time']) / previous_summary['average_execution_time']) * 100
            improvements.append(f"Execution time improved by {improvement_percent:.1f}%")
        
        # Build report
        report = []
        report.append("## Benchmark Improvement Report")
        report.append("")
        report.append(f"**Current Results:** {len(current_results)} tests")
        report.append(f"**Previous Results:** {len(previous_results)} tests")
        report.append("")
        
        # Summary comparison
        report.append("### Summary Comparison")
        report.append("")
        report.append("| Metric | Current | Previous | Change |")
        report.append("|--------|---------|----------|--------|")
        
        score_change = current_summary['overall_score'] - previous_summary['overall_score']
        score_change_pct = (score_change / previous_summary['overall_score']) * 100
        report.append(f"| Overall Score | {current_summary['overall_score']:.3f} | {previous_summary['overall_score']:.3f} | {score_change:+.3f} ({score_change_pct:+.1f}%) |")
        
        pass_rate_change = current_summary['pass_rate'] - previous_summary['pass_rate']
        pass_rate_change_pct = (pass_rate_change / previous_summary['pass_rate']) * 100
        report.append(f"| Pass Rate | {current_summary['pass_rate']:.1%} | {previous_summary['pass_rate']:.1%} | {pass_rate_change:+.1%} ({pass_rate_change_pct:+.1f}%) |")
        
        time_change = current_summary['average_execution_time'] - previous_summary['average_execution_time']
        time_change_pct = (time_change / previous_summary['average_execution_time']) * 100
        report.append(f"| Avg Execution Time | {current_summary['average_execution_time']:.1f}s | {previous_summary['average_execution_time']:.1f}s | {time_change:+.1f}s ({time_change_pct:+.1f}%) |")
        
        report.append("")
        
        # Improvements
        if improvements:
            report.append("### Improvements")
            report.append("")
            for improvement in improvements:
                report.append(f"- ✅ {improvement}")
            report.append("")
        
        # Regressions
        if regressions:
            report.append("### Regressions")
            report.append("")
            for regression in regressions:
                severity_icon = {"low": "⚠️", "medium": "🔶", "high": "🔴", "critical": "🚨"}.get(regression.severity, "❓")
                report.append(f"- {severity_icon} **{regression.test_id}** ({regression.type}): {regression.description}")
            report.append("")
        
        # Overall status
        if regressions:
            critical_regressions = [r for r in regressions if r.severity == "critical"]
            if critical_regressions:
                report.append("### Status: 🚨 CRITICAL REGRESSIONS DETECTED")
            else:
                report.append("### Status: ⚠️ REGRESSIONS DETECTED")
        elif improvements:
            report.append("### Status: ✅ IMPROVEMENTS DETECTED")
        else:
            report.append("### Status: ➡️ STABLE")
        
        return "\n".join(report)
    
    def _generate_summary_stats(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Generate summary statistics from results."""
        if not results:
            return {
                "total_tests": 0,
                "passed_tests": 0,
                "failed_tests": 0,
                "pass_rate": 0.0,
                "overall_score": 0.0,
                "average_execution_time": 0.0
            }
        
        total_tests = len(results)
        passed_tests = sum(1 for r in results if r.passed)
        failed_tests = total_tests - passed_tests
        pass_rate = passed_tests / total_tests if total_tests > 0 else 0.0
        overall_score = statistics.mean([r.overall_score for r in results]) if results else 0.0
        avg_execution_time = statistics.mean([r.execution_time for r in results]) if results else 0.0
        
        return {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "pass_rate": pass_rate,
            "overall_score": overall_score,
            "average_execution_time": avg_execution_time
        }
    
    def _calculate_severity(self, good_value: float, bad_value: float) -> str:
        """Calculate severity of regression based on percentage change."""
        if good_value == 0:
            return "critical"
        
        change_percent = abs((bad_value - good_value) / good_value) * 100
        
        if change_percent >= 50:
            return "critical"
        elif change_percent >= 25:
            return "high"
        elif change_percent >= 10:
            return "medium"
        else:
            return "low"
    
    def list_available_baselines(self) -> List[Dict[str, Any]]:
        """List all available baseline versions."""
        baseline_files = list(self.baselines_dir.glob("baseline_*.json"))
        baselines = []
        
        for baseline_file in baseline_files:
            try:
                with open(baseline_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                baselines.append({
                    "file": baseline_file.name,
                    "version": data.get("version", "unknown"),
                    "timestamp": data.get("timestamp", "unknown"),
                    "total_tests": data.get("total_tests", 0),
                    "file_size": baseline_file.stat().st_size,
                    "modified": datetime.fromtimestamp(baseline_file.stat().st_mtime).isoformat()
                })
            except Exception as e:
                self.logger.warning(f"Failed to read baseline file {baseline_file}: {e}")
        
        # Sort by modification time (newest first)
        baselines.sort(key=lambda x: x["modified"], reverse=True)
        return baselines
