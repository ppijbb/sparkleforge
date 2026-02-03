#!/usr/bin/env python3
"""
Calculate AgentBench scores (Web Navigation, Tool Usage, Multi-Agent, Reasoning)
from actual benchmark results and update README.md
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).parent))

from benchmark_runner import BenchmarkRunner
from agent_benchmark_metrics import AgentBenchmarkAnalyzer

def calculate_agentbench_scores(results_dir: str = "results") -> Dict[str, float]:
    """Calculate AgentBench category scores from benchmark results."""
    
    results_path = project_root / results_dir
    if not results_path.exists():
        print(f"⚠️  Results directory not found: {results_path}")
        print("   Running benchmarks first...")
        return None
    
    # Find latest results file
    json_files = list(results_path.glob("benchmark_results_*.json"))
    if not json_files:
        print(f"⚠️  No benchmark results found in {results_path}")
        print("   Run benchmarks first with: python run_benchmarks.py")
        return None
    
    latest_file = max(json_files, key=lambda p: p.stat().st_mtime)
    print(f"📊 Loading results from: {latest_file}")
    
    with open(latest_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Group results by category
    category_scores = defaultdict(list)
    
    for result in data.get('results', []):
        category = result.get('category', 'unknown')
        overall_score = result.get('overall_score', 0.0)
        
        # Map categories to AgentBench categories
        if category == 'WebNavigation':
            category_scores['web_navigation'].append(overall_score)
        elif category == 'ToolUsage':
            category_scores['tool_usage'].append(overall_score)
        elif category == 'MultiAgent':
            category_scores['multi_agent'].append(overall_score)
        elif category == 'Reasoning':
            category_scores['reasoning'].append(overall_score)
    
    # Calculate average scores per category
    agentbench_scores = {}
    for category, scores in category_scores.items():
        if scores:
            avg_score = sum(scores) / len(scores)
            agentbench_scores[category] = avg_score * 100  # Convert to percentage
        else:
            agentbench_scores[category] = 0.0
    
    # Calculate overall score
    if agentbench_scores:
        # Weighted average (same weights as AgentBenchmarkAnalyzer)
        weights = {
            'web_navigation': 0.25,
            'tool_usage': 0.25,
            'multi_agent': 0.25,
            'reasoning': 0.15,
            'agent_performance': 0.10
        }
        
        weighted_sum = sum(agentbench_scores.get(cat, 0) * weights.get(cat, 0) 
                          for cat in ['web_navigation', 'tool_usage', 'multi_agent', 'reasoning'])
        total_weight = sum(weights.get(cat, 0) 
                          for cat in ['web_navigation', 'tool_usage', 'multi_agent', 'reasoning'])
        
        overall = weighted_sum / total_weight if total_weight > 0 else 0.0
        agentbench_scores['overall'] = overall
    else:
        agentbench_scores['overall'] = 0.0
    
    return agentbench_scores


def run_benchmarks_and_calculate():
    """Run benchmarks and calculate AgentBench scores."""
    print("=" * 80)
    print("🚀 Running AgentBench Evaluation")
    print("=" * 80)
    print()
    
    try:
        # Initialize benchmark runner
        config_path = project_root / "tests" / "benchmark" / "benchmark_config.yaml"
        thresholds_path = project_root / "tests" / "benchmark" / "benchmark_thresholds.yaml"
        
        runner = BenchmarkRunner(
            str(project_root),
            str(config_path),
            str(thresholds_path)
        )
        
        print("✅ BenchmarkRunner initialized")
        print()
        
        # Run all benchmarks
        print("📊 Running agent evaluation benchmarks...")
        results = runner.run_all_benchmarks()
        
        if not results:
            print("❌ No results generated")
            return None
        
        print(f"✅ Completed {len(results)} benchmark tasks")
        print()
        
        # Calculate category scores
        category_scores = defaultdict(list)
        category_names = {
            'WebNavigation': 'Web Navigation',
            'ToolUsage': 'Tool Usage',
            'MultiAgent': 'Multi-Agent',
            'Reasoning': 'Reasoning'
        }
        
        for result in results:
            category = result.category
            if category in category_names:
                category_scores[category].append(result.overall_score)
        
        # Calculate averages
        agentbench_scores = {}
        for category, scores in category_scores.items():
            if scores:
                avg_score = sum(scores) / len(scores)
                agentbench_scores[category_names[category]] = avg_score * 100
        
        # Calculate overall
        if agentbench_scores:
            overall = sum(agentbench_scores.values()) / len(agentbench_scores)
            agentbench_scores['Overall'] = overall
        
        return agentbench_scores
        
    except Exception as e:
        print(f"❌ Error running benchmarks: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Try to load existing results first
    scores = calculate_agentbench_scores()
    
    if scores is None:
        # Run benchmarks if no results found
        scores = run_benchmarks_and_calculate()
    
    if scores:
        print("=" * 80)
        print("📊 AgentBench Scores")
        print("=" * 80)
        print()
        
        for category, score in scores.items():
            print(f"  {category}: {score:.1f}%")
        
        print()
        print("=" * 80)
        
        # Save to JSON
        output_file = project_root / "results" / "agentbench_scores.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(scores, f, indent=2)
        
        print(f"💾 Scores saved to: {output_file}")
    else:
        print("❌ Failed to calculate AgentBench scores")
        sys.exit(1)

