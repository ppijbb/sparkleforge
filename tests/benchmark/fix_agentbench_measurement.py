#!/usr/bin/env python3
"""
AgentBench 측정 방법 개선 - 실제 결과 구조에 맞게 수정
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).parent))

from benchmark_runner import BenchmarkRunner


def analyze_result_structure():
    """실제 실행 결과 구조 분석"""
    print("=" * 80)
    print("🔍 결과 구조 분석")
    print("=" * 80)
    print()
    
    runner = BenchmarkRunner(
        str(project_root),
        str(project_root / "tests" / "benchmark" / "benchmark_config.yaml"),
        str(project_root / "tests" / "benchmark" / "benchmark_thresholds.yaml")
    )
    
    # 첫 번째 태스크 실행
    agent_tasks = runner.config.get('agent_tasks', [])
    if not agent_tasks:
        print("❌ No agent tasks found")
        return
    
    test_task = agent_tasks[0]
    print(f"📊 테스트: {test_task.get('id')} - {test_task.get('query')}")
    print()
    
    # CLI 실행
    print("⏳ CLI 실행 중...")
    cli_result = runner.cli_executor.execute_research(test_task['query'])
    
    print(f"✅ 실행 완료: {cli_result.success}")
    print(f"   Execution Time: {cli_result.execution_time:.2f}s")
    print()
    
    if cli_result.parsed_output:
        print("📋 결과 구조:")
        print(json.dumps(cli_result.parsed_output, indent=2, default=str)[:1000])
        print("...")
        print()
        
        # 메트릭 추출 테스트
        print("🔍 메트릭 추출 테스트:")
        extracted = runner._extract_agent_metrics_from_output(cli_result.parsed_output)
        print(json.dumps(extracted, indent=2, default=str))
        print()
        
        # 메트릭 수집 테스트
        print("📊 메트릭 수집 테스트:")
        metrics = runner._collect_agent_metrics_for_task(test_task, extracted, 0.0)
        print(f"   Collected {len(metrics)} metrics")
        for metric in metrics[:5]:
            print(f"   - {metric.name}: {metric.value:.2%} (category: {metric.category})")
        print()
        
        # 점수 계산 테스트
        if metrics:
            score = runner._calculate_test_score(metrics)
            print(f"📊 Overall Score: {score:.2%}")
    else:
        print("❌ No parsed output")
        if cli_result.error_message:
            print(f"   Error: {cli_result.error_message}")
        if cli_result.stdout:
            print(f"   Stdout: {cli_result.stdout[:500]}")
        if cli_result.stderr:
            print(f"   Stderr: {cli_result.stderr[:500]}")


def fix_metric_extraction():
    """메트릭 추출 로직 개선 제안"""
    print()
    print("=" * 80)
    print("💡 메트릭 추출 개선 제안")
    print("=" * 80)
    print()
    print("문제점:")
    print("1. CLI 실행 결과가 기대하는 구조와 다를 수 있음")
    print("2. navigation_log, tool_usage_log 등이 실제 결과에 없을 수 있음")
    print("3. execution_results, sources 등의 실제 필드를 사용해야 함")
    print()
    print("개선 방안:")
    print("1. 실제 결과 구조에 맞게 메트릭 추출 로직 수정")
    print("2. execution_results에서 도구 사용, 병렬 실행 등 추출")
    print("3. sources에서 웹 네비게이션 성공 여부 추출")
    print("4. planned_tasks에서 논리적 계획 추출")
    print()


if __name__ == "__main__":
    analyze_result_structure()
    fix_metric_extraction()

