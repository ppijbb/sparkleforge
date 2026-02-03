#!/usr/bin/env python3
"""
SparkleForge Benchmark Runner

Main entry point for running comprehensive benchmark tests.
Supports running all benchmarks, specific suites, baseline comparison, and report generation.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional
import json

try:
    from .benchmark_runner import BenchmarkRunner
    from .benchmark_reporter import BenchmarkReporter
    from .benchmark_comparator import BenchmarkComparator
except ImportError:
    from benchmark_runner import BenchmarkRunner
    from benchmark_reporter import BenchmarkReporter
    from benchmark_comparator import BenchmarkComparator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point for benchmark runner."""
    parser = argparse.ArgumentParser(
        description="Run SparkleForge benchmark tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run comprehensive benchmarks (all metrics in single execution)
  python run_benchmarks.py

  # Run with custom config
  python run_benchmarks.py --config benchmark/benchmark_config.yaml

  # Compare with baseline
  python run_benchmarks.py --compare-baseline v1.0.0

  # Generate report only
  python run_benchmarks.py --report-only --results results/latest.json

  # Run with custom output directory
  python run_benchmarks.py --output-dir custom_results
        """
    )
    
    # Configuration arguments
    parser.add_argument(
        "--config",
        default="benchmark/benchmark_config.yaml",
        help="Path to benchmark configuration file (default: benchmark/benchmark_config.yaml)"
    )
    parser.add_argument(
        "--thresholds",
        default="benchmark/benchmark_thresholds.yaml",
        help="Path to benchmark thresholds file (default: benchmark/benchmark_thresholds.yaml)"
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Output directory for results and reports (default: results)"
    )
    
    # Execution arguments
    parser.add_argument(
        "--suite",
        choices=["research_quality", "performance", "source_validation", 
                "creative_insights", "memory_learning", "all"],
        default="all",
        help="Specific benchmark suite to run (default: all)"
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout per test case in seconds (default: 300)"
    )
    
    # Comparison arguments
    parser.add_argument(
        "--compare-baseline",
        help="Compare with specific baseline version"
    )
    parser.add_argument(
        "--save-baseline",
        help="Save current results as baseline with specified version"
    )
    parser.add_argument(
        "--list-baselines",
        action="store_true",
        help="List available baseline versions"
    )
    
    # Reporting arguments
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Generate reports only (don't run benchmarks)"
    )
    parser.add_argument(
        "--results",
        help="Path to existing results file for report generation"
    )
    parser.add_argument(
        "--format",
        choices=["json", "markdown", "console", "all"],
        default="all",
        help="Report format to generate (default: all)"
    )
    
    # Output arguments
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress non-error output"
    )
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.quiet:
        logging.getLogger().setLevel(logging.ERROR)
    
    # Validate arguments
    if args.report_only and not args.results:
        logger.error("--results is required when using --report-only")
        sys.exit(1)
    
    if args.parallel < 1:
        logger.error("--parallel must be >= 1")
        sys.exit(1)
    
    if args.timeout < 1:
        logger.error("--timeout must be >= 1")
        sys.exit(1)
    
    # Get project root
    project_root = Path(__file__).parent.parent.parent
    config_path = project_root / "tests" / "benchmark" / "benchmark_config.yaml"
    thresholds_path = project_root / "tests" / "benchmark" / "benchmark_thresholds.yaml"
    
    # Validate configuration files
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        sys.exit(1)
    
    if not thresholds_path.exists():
        logger.error(f"Thresholds file not found: {thresholds_path}")
        sys.exit(1)
    
    try:
        # Initialize components
        runner = BenchmarkRunner(str(project_root), str(config_path), str(thresholds_path))
        reporter = BenchmarkReporter(args.output_dir)
        comparator = BenchmarkComparator()
        
        # Handle list baselines
        if args.list_baselines:
            list_baselines(comparator)
            return
        
        # Handle report only mode
        if args.report_only:
            generate_reports_only(reporter, args.results, args.format)
            return
        
        # Run benchmarks
        results = run_benchmarks(runner, args.suite, args.parallel, args.timeout)
        
        if not results:
            logger.error("No benchmark results generated")
            sys.exit(1)
        
        # Generate reports
        generate_reports(reporter, results, args.format, args.output_dir)
        
        # Handle baseline operations
        if args.save_baseline:
            save_baseline(comparator, results, args.save_baseline)
        
        if args.compare_baseline:
            compare_with_baseline(comparator, results, args.compare_baseline)
        
        # Print summary
        print_summary(results)
        
    except KeyboardInterrupt:
        logger.info("Benchmark execution interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Benchmark execution failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def run_benchmarks(runner: BenchmarkRunner, suite: str, parallel: int, timeout: int) -> List:
    """Run benchmark tests based on specified suite with production validation."""
    logger.info(f"Running {suite} benchmark suite...")
    
    # Validate production environment before running benchmarks
    is_valid, issues = runner.cli_executor.validate_environment()
    if not is_valid:
        logger.error(f"Production environment validation failed: {issues}")
        logger.error("Please ensure:")
        logger.error("1. OPENROUTER_API_KEY is set")
        logger.error("2. MCP server is running and configured")
        logger.error("3. All production dependencies are installed")
        logger.error("4. .env file has production settings (no fallbacks)")
        return []
    
    logger.info("✅ Production environment validated successfully")
    
    # Update parallel workers in runner config
    if hasattr(runner, 'config') and 'execution' in runner.config:
        runner.config['execution']['parallel_workers'] = parallel
        runner.config['execution']['timeout'] = timeout
    
    if suite == "all":
        results = runner.run_all_benchmarks()
    elif suite == "research_quality":
        results = runner.run_research_quality_benchmark()
    elif suite == "performance":
        results = runner.run_performance_benchmark()
    elif suite == "source_validation":
        results = runner.run_source_validation_benchmark()
    elif suite == "creative_insights":
        results = runner.run_creative_insights_benchmark()
    elif suite == "memory_learning":
        results = runner.run_memory_learning_benchmark()
    else:
        logger.error(f"Unknown benchmark suite: {suite}")
        return []
    
    logger.info(f"Completed {len(results)} benchmark tests")
    return results


def generate_reports(reporter: BenchmarkReporter, results: List, format: str, output_dir: str) -> None:
    """Generate benchmark reports."""
    logger.info("Generating benchmark reports...")
    
    if format in ["json", "all"]:
        json_file = reporter.generate_json_report(results)
        logger.info(f"JSON report: {json_file}")
    
    if format in ["markdown", "all"]:
        markdown_file = reporter.generate_markdown_report(results)
        logger.info(f"Markdown report: {markdown_file}")
    
    if format in ["console", "all"]:
        console_summary = reporter.generate_console_summary(results)
        print(console_summary)
    
    # Generate charts
    if format in ["markdown", "all"]:
        chart_paths = reporter._generate_charts(results)
        for chart_type, chart_path in chart_paths.items():
            if chart_path:
                logger.info(f"Chart generated: {chart_path}")


def generate_reports_only(reporter: BenchmarkReporter, results_file: str, format: str) -> None:
    """Generate reports from existing results file."""
    logger.info(f"Loading results from {results_file}")
    
    try:
        with open(results_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Convert back to BenchmarkResult objects
        results = []
        for result_data in data.get('results', []):
            from benchmark_metrics import BenchmarkResult
            from datetime import datetime
            
            result = BenchmarkResult(
                test_id=result_data['test_id'],
                category=result_data['category'],
                query=result_data['query'],
                execution_time=result_data['execution_time'],
                metrics=[],  # Simplified for report generation
                overall_score=result_data['overall_score'],
                passed=result_data['passed'],
                timestamp=datetime.fromisoformat(result_data['timestamp']) if result_data.get('timestamp') else None
            )
            results.append(result)
        
        generate_reports(reporter, results, format, "results")
        
    except Exception as e:
        logger.error(f"Failed to load results file: {e}")
        sys.exit(1)


def save_baseline(comparator: BenchmarkComparator, results: List, version: str) -> None:
    """Save current results as baseline."""
    logger.info(f"Saving baseline version: {version}")
    
    baseline_file = comparator.save_baseline(results, version)
    if baseline_file:
        logger.info(f"Baseline saved: {baseline_file}")
    else:
        logger.error("Failed to save baseline")
        sys.exit(1)


def compare_with_baseline(comparator: BenchmarkComparator, results: List, version: str) -> None:
    """Compare current results with baseline."""
    logger.info(f"Comparing with baseline version: {version}")
    
    previous_results = comparator.load_historical_results(version)
    if not previous_results:
        logger.error(f"No baseline found for version: {version}")
        sys.exit(1)
    
    # Detect regressions
    regressions = comparator.detect_regression(results, previous_results)
    
    if regressions:
        logger.warning(f"Found {len(regressions)} regressions:")
        for regression in regressions:
            logger.warning(f"  {regression.test_id}: {regression.description}")
    else:
        logger.info("No regressions detected")
    
    # Generate improvement report
    improvement_report = comparator.generate_improvement_report(results, previous_results)
    print(improvement_report)


def list_baselines(comparator: BenchmarkComparator) -> None:
    """List available baseline versions."""
    baselines = comparator.list_available_baselines()
    
    if not baselines:
        print("No baselines available")
        return
    
    print("Available baselines:")
    print("-" * 80)
    print(f"{'Version':<20} {'Tests':<8} {'Modified':<20} {'File'}")
    print("-" * 80)
    
    for baseline in baselines:
        print(f"{baseline['version']:<20} {baseline['total_tests']:<8} {baseline['modified'][:19]:<20} {baseline['file']}")


def print_summary(results: List) -> None:
    """Print benchmark execution summary."""
    if not results:
        return
    
    total_tests = len(results)
    passed_tests = sum(1 for r in results if r.passed)
    failed_tests = total_tests - passed_tests
    pass_rate = passed_tests / total_tests if total_tests > 0 else 0.0
    
    print("\n" + "=" * 60)
    print("BENCHMARK EXECUTION SUMMARY")
    print("=" * 60)
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests} ({pass_rate:.1%})")
    print(f"Failed: {failed_tests} ({1-pass_rate:.1%})")
    
    if failed_tests > 0:
        print("\nFailed Tests:")
        for result in results:
            if not result.passed:
                print(f"  - {result.test_id} ({result.category})")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
