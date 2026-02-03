#!/usr/bin/env python3
"""
AgentBench 점수 개선 분석 - README의 값이 실제 측정값인지 확인하고 개선 여부 분석
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent

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

print("📋 Current README Values:")
for category, score in readme_scores.items():
    print(f"  {category}: {score}%")
print()

# 실제 측정 결과 확인
results_file = project_root / "results" / "agentbench_scores.json"
if results_file.exists():
    import json
    with open(results_file, 'r') as f:
        actual_scores = json.load(f)
    
    print("📊 Actual Measurement Results:")
    for category, score in actual_scores.items():
        print(f"  {category}: {score}%")
    print()
    
    # 비교
    print("📈 Comparison:")
    for category in readme_scores.keys():
        readme_val = readme_scores.get(category, 0)
        actual_val = actual_scores.get(category, 0)
        
        if actual_val > 0:
            diff = actual_val - readme_val
            if diff > 0:
                print(f"  ✅ {category}: {readme_val}% → {actual_val}% (+{diff:.1f}%)")
            elif diff < 0:
                print(f"  ⚠️  {category}: {readme_val}% → {actual_val}% ({diff:.1f}%)")
            else:
                print(f"  ➡️  {category}: {readme_val}% (unchanged)")
        else:
            print(f"  ❌ {category}: No actual measurement available")
else:
    print("⚠️  No actual measurement results found")
    print("   Current README values may be:")
    print("   - Placeholder values")
    print("   - From previous benchmark runs")
    print("   - Estimated values")
    print()
    print("   To get actual measurements, run:")
    print("   python tests/benchmark/run_benchmarks.py")

print()
print("=" * 80)
print()
print("💡 Note: README의 58.0%, 59.0%, 59.5%, 56.8%, 58.3% 값들은")
print("   실제 벤치마크 실행 결과가 아니라면")
print("   실제 측정을 통해 업데이트해야 합니다.")
print()
print("=" * 80)

