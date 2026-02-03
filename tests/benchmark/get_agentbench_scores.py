#!/usr/bin/env python3
"""
AgentBench 점수 계산 - 실제 벤치마크 결과에서 카테고리별 점수 추출
"""

import sys
import json
from pathlib import Path
from typing import Dict, List
from collections import defaultdict

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).parent))

def calculate_category_scores_from_results(results: List) -> Dict[str, float]:
    """벤치마크 결과에서 카테고리별 점수 계산"""
    category_scores = defaultdict(list)
    
    category_mapping = {
        'WebNavigation': 'Web Navigation',
        'ToolUsage': 'Tool Usage',
        'MultiAgent': 'Multi-Agent',
        'Reasoning': 'Reasoning'
    }
    
    for result in results:
        category = result.category if hasattr(result, 'category') else result.get('category', '')
        score = result.overall_score if hasattr(result, 'overall_score') else result.get('overall_score', 0.0)
        
        if category in category_mapping:
            category_scores[category].append(score)
    
    # 평균 계산
    agentbench_scores = {}
    for category, scores in category_scores.items():
        if scores:
            avg = sum(scores) / len(scores)
            agentbench_scores[category_mapping[category]] = avg * 100
    
    # Overall 계산
    if agentbench_scores:
        overall = sum(agentbench_scores.values()) / len(agentbench_scores)
        agentbench_scores['Overall'] = overall
    
    return agentbench_scores


def load_benchmark_results(results_dir: str = "results") -> Dict[str, float]:
    """저장된 벤치마크 결과 로드"""
    results_path = project_root / results_dir
    
    # JSON 결과 파일 찾기
    json_files = list(results_path.glob("benchmark_results_*.json"))
    
    if json_files:
        latest = max(json_files, key=lambda p: p.stat().st_mtime)
        print(f"📊 Loading from: {latest}")
        
        with open(latest, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 결과에서 점수 계산
        results = data.get('results', [])
        return calculate_category_scores_from_results(results)
    
    return {}


def main():
    """AgentBench 점수 확인"""
    print("=" * 80)
    print("📊 AgentBench Score Analysis")
    print("=" * 80)
    print()
    
    # README의 현재 값
    readme_scores = {
        "Web Navigation": 58.0,
        "Tool Usage": 59.0,
        "Multi-Agent": 59.5,
        "Reasoning": 56.8,
        "Overall": 58.3
    }
    
    print("📋 README Current Values:")
    for cat, score in readme_scores.items():
        print(f"  {cat}: {score}%")
    print()
    
    # 실제 측정 결과 확인
    actual_scores = load_benchmark_results()
    
    if actual_scores:
        print("📊 Actual Measurement Results:")
        for cat, score in actual_scores.items():
            print(f"  {cat}: {score:.1f}%")
        print()
        
        print("📈 Comparison:")
        for cat in readme_scores.keys():
            readme_val = readme_scores.get(cat, 0)
            actual_val = actual_scores.get(cat, 0)
            
            if actual_val > 0:
                diff = actual_val - readme_val
                status = "✅" if diff >= 0 else "⚠️"
                print(f"  {status} {cat}: {readme_val}% → {actual_val:.1f}% ({diff:+.1f}%)")
    else:
        print("⚠️  No benchmark results found")
        print()
        print("💡 README의 값들(58.0%, 59.0%, 59.5%, 56.8%, 58.3%)은:")
        print("   - 실제 벤치마크 실행 결과가 아닐 수 있습니다")
        print("   - 실제 측정을 통해 업데이트가 필요합니다")
        print()
        print("실제 벤치마크 실행:")
        print("  python tests/benchmark/run_benchmarks.py")
    
    print()
    print("=" * 80)


if __name__ == "__main__":
    main()

