#!/usr/bin/env python3
"""
실제 AgentBench 점수 확인 - README의 값이 실제 측정값인지 확인
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
readme_path = project_root / "README.md"

print("=" * 80)
print("📊 AgentBench Scores in README.md")
print("=" * 80)
print()

with open(readme_path, 'r', encoding='utf-8') as f:
    content = f.read()
    
    # SparkleForge 행 찾기
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if 'SparkleForge' in line and 'AgentBench' in line:
            print(f"Line {i+1}: {line.strip()}")
            print()
            
            # 다음 몇 줄 확인
            for j in range(i, min(i+3, len(lines))):
                print(f"  {lines[j]}")
            break

print()
print("=" * 80)
print("⚠️  현재 README의 값들:")
print("  - Web Navigation: 58.0%")
print("  - Tool Usage: 59.0%")
print("  - Multi-Agent: 59.5%")
print("  - Reasoning: 56.8%")
print("  - Overall: 58.3%")
print()
print("이 값들은 실제 벤치마크 실행 결과가 아니라면")
print("실제 측정이 필요합니다.")
print("=" * 80)

