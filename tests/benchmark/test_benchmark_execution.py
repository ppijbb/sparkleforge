#!/usr/bin/env python3
"""
벤치마크 실행 테스트 - 실제 실행 가능 여부 확인
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).parent))

from benchmark_runner import BenchmarkRunner

def test_benchmark_runner_init():
    """BenchmarkRunner 초기화 테스트"""
    print("=" * 80)
    print("🔍 Benchmark Runner 초기화 테스트")
    print("=" * 80)
    print()
    
    try:
        config_path = project_root / "tests" / "benchmark" / "benchmark_config.yaml"
        thresholds_path = project_root / "benchmark" / "benchmark_thresholds.yaml"
        
        if not thresholds_path.exists():
            thresholds_path = project_root / "tests" / "benchmark" / "benchmark_thresholds.yaml"
        
        runner = BenchmarkRunner(
            str(project_root),
            str(config_path),
            str(thresholds_path)
        )
        
        print("✅ BenchmarkRunner 초기화 성공")
        print(f"   Config 파일: {config_path}")
        print(f"   Thresholds 파일: {thresholds_path}")
        print()
        
        # Config 확인
        agent_tasks = runner.config.get('agent_tasks', [])
        print(f"📋 Agent Tasks: {len(agent_tasks)}개")
        for task in agent_tasks[:3]:  # 처음 3개만
            print(f"   - {task.get('id')}: {task.get('category')} - {task.get('query')[:50]}...")
        print()
        
        # CLI Executor 확인
        print("🔧 CLI Executor 확인:")
        env_valid, issues = runner.cli_executor.validate_environment()
        if env_valid:
            print("   ✅ 환경 검증 통과")
        else:
            print(f"   ❌ 환경 검증 실패: {issues}")
        print()
        
        return runner
        
    except Exception as e:
        print(f"❌ 초기화 실패: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_single_task_execution(runner):
    """단일 태스크 실행 테스트"""
    print("=" * 80)
    print("🚀 단일 태스크 실행 테스트")
    print("=" * 80)
    print()
    
    if not runner:
        print("❌ Runner가 없습니다")
        return
    
    agent_tasks = runner.config.get('agent_tasks', [])
    if not agent_tasks:
        print("❌ Agent tasks가 없습니다")
        return
    
    # 첫 번째 태스크만 테스트
    test_task = agent_tasks[0]
    print(f"📊 테스트 태스크: {test_task.get('id')}")
    print(f"   Category: {test_task.get('category')}")
    print(f"   Query: {test_task.get('query')}")
    print()
    
    try:
        print("⏳ 실행 중...")
        result = runner._run_single_agent_task_comprehensive(test_task)
        
        print(f"✅ 실행 완료")
        print(f"   Test ID: {result.test_id}")
        print(f"   Category: {result.category}")
        print(f"   Execution Time: {result.execution_time:.2f}s")
        print(f"   Overall Score: {result.overall_score:.2%}")
        print(f"   Passed: {result.passed}")
        print(f"   Metrics: {len(result.metrics)}개")
        
        if result.metrics:
            print()
            print("📊 메트릭:")
            for metric in result.metrics[:5]:  # 처음 5개만
                print(f"   - {metric.name}: {metric.value:.2%} (threshold: {metric.threshold:.2%}, passed: {metric.passed})")
        
        return result
        
    except Exception as e:
        print(f"❌ 실행 실패: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_score_calculation(runner):
    """점수 계산 테스트"""
    print()
    print("=" * 80)
    print("📊 점수 계산 테스트")
    print("=" * 80)
    print()
    
    if not runner:
        return
    
    agent_tasks = runner.config.get('agent_tasks', [])
    
    category_scores = {}
    category_counts = {}
    
    for task in agent_tasks:
        category = task.get('category', 'Unknown')
        
        try:
            result = runner._run_single_agent_task_comprehensive(task)
            
            if category not in category_scores:
                category_scores[category] = []
                category_counts[category] = 0
            
            category_scores[category].append(result.overall_score)
            category_counts[category] += 1
            
            print(f"✅ {task.get('id')}: {result.overall_score:.2%}")
            
        except Exception as e:
            print(f"❌ {task.get('id')}: 실패 - {e}")
    
    print()
    print("📊 카테고리별 평균 점수:")
    category_mapping = {
        'WebNavigation': 'Web Navigation',
        'ToolUsage': 'Tool Usage',
        'MultiAgent': 'Multi-Agent',
        'Reasoning': 'Reasoning'
    }
    
    for category, scores in category_scores.items():
        if scores:
            avg = sum(scores) / len(scores)
            name = category_mapping.get(category, category)
            print(f"   {name}: {avg:.2%} ({len(scores)}개 테스트)")
    
    # Overall 계산
    all_scores = []
    for scores in category_scores.values():
        all_scores.extend(scores)
    
    if all_scores:
        overall = sum(all_scores) / len(all_scores)
        print()
        print(f"📊 Overall Score: {overall:.2%}")
        
        return {
            'Web Navigation': category_scores.get('WebNavigation', [0])[0] * 100 if category_scores.get('WebNavigation') else 0,
            'Tool Usage': category_scores.get('ToolUsage', [0])[0] * 100 if category_scores.get('ToolUsage') else 0,
            'Multi-Agent': category_scores.get('MultiAgent', [0])[0] * 100 if category_scores.get('MultiAgent') else 0,
            'Reasoning': category_scores.get('Reasoning', [0])[0] * 100 if category_scores.get('Reasoning') else 0,
            'Overall': overall * 100
        }
    
    return None


def main():
    """메인 테스트 실행"""
    print("=" * 80)
    print("🧪 벤치마크 실행 및 측정 방법 점검")
    print("=" * 80)
    print()
    
    # 1. 초기화 테스트
    runner = test_benchmark_runner_init()
    
    if not runner:
        print("❌ 초기화 실패로 테스트 중단")
        return
    
    # 2. 단일 태스크 실행 테스트
    test_single_task_execution(runner)
    
    print()
    print("=" * 80)
    print("💡 전체 벤치마크 실행하려면:")
    print("   python tests/benchmark/run_benchmarks.py")
    print("=" * 80)


if __name__ == "__main__":
    main()

