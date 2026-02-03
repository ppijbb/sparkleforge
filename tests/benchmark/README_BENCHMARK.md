# SparkleForge Agent Evaluation Benchmark System

A comprehensive, production-ready agent evaluation benchmark system based on academic standards (WebArena, ToolBench, AgentBench, ALFWorld) for evaluating SparkleForge agent performance across web navigation, tool usage, multi-agent collaboration, reasoning, and planning capabilities. **No dummy data, no fallbacks, no mock code** - only real agent performance measurements using OpenRouter + Gemini 2.5 Flash Lite.

## 🎯 Overview

The SparkleForge agent evaluation benchmark system provides **comprehensive agent measurement** - all agent metrics are collected in a single execution for maximum efficiency and accuracy:

- **Single Execution**: One CLI run measures all agent capabilities (web navigation, tool usage, multi-agent collaboration, reasoning, planning)
- **No Redundancy**: Eliminates duplicate API calls and repeated processing
- **Complete Coverage**: Every agent task generates metrics for all evaluation categories
- **Academic Standards**: Based on WebArena, ToolBench, AgentBench, ALFWorld standards

The SparkleForge agent evaluation benchmark system provides:

- **Comprehensive Agent Testing**: Evaluates all agent capabilities across 5 categories
- **Reproducible Results**: Predefined agent tasks with expected outcomes
- **Multiple Output Formats**: JSON, Markdown, and console reports
- **Regression Detection**: Compare against historical baselines
- **CI/CD Integration**: Pytest integration and automated reporting
- **Real-time Monitoring**: Live progress tracking and performance metrics

## 📊 Agent Evaluation Categories

### 1. Web Navigation (WebArena-style)
- **Navigation Success Rate**: 80% successful web navigation
- **Information Accuracy**: 85% accurate information retrieval
- **Page Load Success Rate**: 90% successful page loads
- **Search Query Effectiveness**: 80% effective search queries

### 2. Tool Usage (ToolBench-style)
- **Tool Usage Success Rate**: 85% successful tool usage
- **Tool Coordination Efficiency**: 80% efficient tool coordination
- **Output Quality**: 80% high-quality outputs
- **Error Recovery Rate**: 70% successful error recovery

### 3. Multi-Agent Collaboration (AgentBench-style)
- **Coordination Efficiency**: 80% efficient agent coordination
- **Task Completion Rate**: 90% task completion rate
- **Result Consistency**: 85% consistent results across agents
- **Conflict Resolution Rate**: 80% successful conflict resolution

### 4. Reasoning and Planning (ALFWorld-style)
- **Reasoning Accuracy**: 90% accurate logical reasoning
- **Logical Consistency**: 95% logical consistency
- **Conclusion Validity**: 85% valid conclusions
- **Plan Feasibility**: 90% feasible plans

### 5. Agent Performance
- **Execution Efficiency**: 80% execution efficiency
- **Resource Utilization**: 70% optimal resource utilization
- **Scalability Score**: 80% scalability performance
- **Reliability Score**: 90% system reliability

## 🚀 Quick Start

### Prerequisites
- **OpenRouter API Key**: Required for Gemini 2.5 Flash Lite access
- **MCP Server**: Must be running and properly configured
- **Production Dependencies**: All required packages installed
- **No Fallbacks**: Environment must be configured for production use

### 1. Run Comprehensive Benchmarks (All Metrics)
```bash
cd /home/user/workspace/mcp_agent/local_researcher_project
python tests/benchmark/run_benchmarks.py
```

**Note**: The system will validate your production environment before running benchmarks. Ensure:
- `OPENROUTER_API_KEY` is set
- MCP server is running
- No fallback modes are enabled
- All production dependencies are installed

### 2. Run with Custom Configuration
```bash
# Use custom config file
python tests/benchmark/run_benchmarks.py --config benchmark/benchmark_config.yaml

# Use custom thresholds
python tests/benchmark/run_benchmarks.py --thresholds benchmark/benchmark_thresholds.yaml

# Custom output directory
python tests/benchmark/run_benchmarks.py --output-dir custom_results
```

### 3. Run with Custom Configuration
```bash
python tests/run_benchmarks.py \
    --config tests/benchmark_config.yaml \
    --thresholds tests/benchmark_thresholds.yaml \
    --output-dir custom_results \
    --parallel 2 \
    --timeout 300
```

### 4. Generate Reports Only
```bash
python tests/run_benchmarks.py \
    --report-only \
    --results results/benchmark_results_20251024_135128.json \
    --format markdown
```

## 📁 File Structure

```
tests/
├── benchmark_config.yaml          # Test cases and configuration
├── benchmark_thresholds.yaml      # Quality thresholds and criteria
├── benchmark_runner.py            # Main benchmark orchestration
├── benchmark_metrics.py           # Metrics collection and analysis
├── benchmark_reporter.py          # Report generation (JSON, Markdown, Console)
├── benchmark_comparator.py        # Baseline comparison and regression detection
├── cli_executor.py               # CLI subprocess execution
├── run_benchmarks.py             # Main entry point script
├── test_benchmark_*.py           # Individual benchmark test suites
├── pytest_benchmark_integration.py # Pytest integration
├── test_benchmark_system.py      # System validation tests
├── baselines/                    # Historical baseline results
└── results/                      # Generated reports and charts
```

## ⚙️ Configuration

### Test Cases (`benchmark_config.yaml`)
```yaml
test_cases:
  - id: "tech-001"
    category: "Technology"
    query: "Latest AI developments in 2025"
    expected:
      min_sources: 8
      min_credibility: 0.7
      min_insights: 2
      max_response_time: 120
```

### Quality Thresholds (`benchmark_thresholds.yaml`)
```yaml
thresholds:
  response_time: 120
  source_credibility: 0.7
  factual_accuracy: 0.85
  creative_novelty: 0.6
  memory_precision: 0.7
```

## 📈 Reports and Output

### JSON Report
```json
{
  "metadata": {
    "timestamp": "2025-10-24T13:51:28",
    "total_tests": 5,
    "version": "2.1.0"
  },
  "summary": {
    "total_tests": 5,
    "passed_tests": 4,
    "failed_tests": 1,
    "pass_rate": 0.8,
    "overall_score": 0.75
  },
  "results": [...],
  "categories": {...},
  "performance": {...}
}
```

### Markdown Report
- Executive summary with key metrics
- Category breakdown with pass rates
- Detailed results for each test
- Visualizations and charts
- Recommendations for improvement

### Console Summary
```
============================================================
SPARKLEFORGE BENCHMARK SUMMARY
============================================================

Total Tests: 5
Passed: 4 (80.0%)
Failed: 1 (20.0%)
Overall Score: 0.750
Average Execution Time: 45.2s

CATEGORY BREAKDOWN:
------------------------------
Technology      |  2/ 2 | 100.0% | 0.800
Science         |  1/ 2 |  50.0% | 0.600
Creative        |  1/ 1 | 100.0% | 0.900
```

## 🔄 Baseline Management

### Save Current Results as Baseline
```bash
python tests/run_benchmarks.py --save-baseline v2.1.0
```

### Compare with Baseline
```bash
python tests/run_benchmarks.py --compare-baseline v2.0.0
```

### List Available Baselines
```bash
python tests/run_benchmarks.py --list-baselines
```

## 🧪 Testing and Validation

### Run System Tests
```bash
python tests/test_benchmark_system.py
```

### Run Pytest Integration
```bash
# Run all benchmark tests
pytest tests/pytest_benchmark_integration.py -v

# Run specific test categories
pytest tests/pytest_benchmark_integration.py -m benchmark -v
pytest tests/pytest_benchmark_integration.py -m slow -v
pytest tests/pytest_benchmark_integration.py -m regression -v
```

### Run Individual Test Suites
```bash
pytest tests/test_benchmark_research_quality.py -v
pytest tests/test_benchmark_performance.py -v
pytest tests/test_benchmark_creative_insights.py -v
```

## 📊 Metrics and Scoring

### Overall Score Calculation
The overall score is calculated as the average of individual test scores:
```
Overall Score = Σ(test_scores) / number_of_tests
```

### Category Scoring
Each category is scored based on relevant metrics:
- **Research Quality**: Source credibility + Factual accuracy
- **Performance**: Response time + Throughput
- **Source Validation**: Credibility + Fact-checking accuracy
- **Creative Insights**: Novelty + Applicability scores
- **Memory Learning**: Precision + Recall rates

### Pass/Fail Criteria
Tests pass when:
- All critical metrics meet thresholds
- Expected values are satisfied
- No critical regressions detected

## 🔧 Advanced Usage

### Custom Test Cases
Add new test cases to `benchmark_config.yaml`:
```yaml
test_cases:
  - id: "custom-001"
    category: "Custom"
    query: "Your custom research query"
    expected:
      min_sources: 10
      min_credibility: 0.8
      min_insights: 3
      max_response_time: 180
```

### Custom Thresholds
Modify thresholds in `benchmark_thresholds.yaml`:
```yaml
thresholds:
  response_time: 60  # More strict
  source_credibility: 0.8  # Higher quality requirement
  creative_novelty: 0.7  # More creative insights required
```

### Parallel Execution
Run benchmarks in parallel for faster execution:
```bash
python tests/run_benchmarks.py --parallel 4
```

### Custom Output Directory
```bash
python tests/run_benchmarks.py --output-dir /path/to/results
```

## 🚨 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure you're running from the project root directory
2. **Configuration Not Found**: Check that config files exist and are valid YAML
3. **CLI Execution Fails**: Verify environment setup and API keys
4. **No Results Generated**: Check timeout settings and test case validity

### Debug Mode
```bash
python tests/run_benchmarks.py --verbose --suite research_quality
```

### Environment Validation
```bash
python -c "from tests.cli_executor import CLIExecutor; CLIExecutor('.').validate_environment()"
```

## 📚 API Reference

### BenchmarkRunner
- `run_all_benchmarks()`: Run all benchmark suites
- `run_research_quality_benchmark()`: Run research quality tests
- `run_performance_benchmark()`: Run performance tests
- `get_benchmark_summary(results)`: Generate summary statistics

### MetricsCollector
- `measure_response_time(start, end)`: Measure execution time
- `measure_source_credibility(sources)`: Evaluate source quality
- `measure_factual_accuracy(claims, sources)`: Verify factual correctness
- `measure_creative_quality(insights)`: Assess creative insights

### BenchmarkReporter
- `generate_json_report(results)`: Create JSON report
- `generate_markdown_report(results)`: Create Markdown report
- `generate_console_summary(results)`: Create console output

### BenchmarkComparator
- `detect_regression(current, previous)`: Find performance regressions
- `compare_with_threshold(current, thresholds)`: Compare against thresholds
- `generate_improvement_report(current, previous)`: Create improvement report

## 🤝 Contributing

### Adding New Metrics
1. Add metric to `MetricsCollector` class
2. Update `BenchmarkRunner._collect_metrics_for_test()`
3. Add threshold to `benchmark_thresholds.yaml`
4. Update test cases in `test_benchmark_*.py`

### Adding New Test Cases
1. Add test case to `benchmark_config.yaml`
2. Update expected values and thresholds
3. Test with `python tests/run_benchmarks.py --suite all`

### Adding New Report Formats
1. Add method to `BenchmarkReporter` class
2. Update `run_benchmarks.py` argument parser
3. Add format option to report generation

## 📄 License

This benchmark system is part of the SparkleForge project and follows the same MIT License.

---

**Last Updated**: October 24, 2025  
**Version**: 2.1.0  
**Status**: Production Ready ✅
