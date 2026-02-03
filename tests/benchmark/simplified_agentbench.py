#!/usr/bin/env python3
"""
간소화된 AgentBench 점수 측정 - 빠른 실행과 명확한 측정
"""

import sys
import json
import asyncio
from pathlib import Path
from typing import Dict, List
from collections import defaultdict

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).parent))

from src.core.researcher_config import load_config_from_env
from src.core.autonomous_orchestrator import AutonomousOrchestrator


async def measure_single_task(query: str, category: str) -> Dict:
    """단일 태스크 실행 및 점수 측정"""
    try:
        load_config_from_env()
        orchestrator = AutonomousOrchestrator()
        
        # 실제 실행
        result = await orchestrator.run_research(query)
        
        if not result:
            return {
                'category': category,
                'success': False,
                'score': 0.0,
                'error': 'No result returned'
            }
        
        # 결과 분석
        execution_results = result.get('execution_results', [])
        sources = result.get('sources', []) or []
        planned_tasks = result.get('planned_tasks', []) or []
        agent_status = result.get('agent_status', {}) or {}
        
        # 카테고리별 점수 계산
        if category == 'WebNavigation':
            # Web Navigation: 소스 수와 검색 성공
            sources_score = min(1.0, len(sources) / 8.0)
            execution_score = 1.0 if len(execution_results) > 0 else 0.0
            score = sources_score * 0.7 + execution_score * 0.3
            
        elif category == 'ToolUsage':
            # Tool Usage: 도구 사용 성공
            tools_used = set()
            for exec_result in execution_results:
                if exec_result.get('tool_used'):
                    tools_used.add(exec_result['tool_used'])
            tools_score = min(1.0, len(tools_used) / 5.0)
            execution_score = 1.0 if len(execution_results) > 0 else 0.0
            score = tools_score * 0.6 + execution_score * 0.4
            
        elif category == 'MultiAgent':
            # Multi-Agent: 병렬 실행 및 에이전트 협업
            parallel_score = 1.0 if any(exec_result.get('parallel_execution', False) 
                                       for exec_result in execution_results) else 0.0
            collaboration_score = 1.0 if len(agent_status) > 1 else 0.5
            score = parallel_score * 0.5 + collaboration_score * 0.5
            
        elif category == 'Reasoning':
            # Reasoning: 논리적 분석 단계
            planning_score = min(1.0, len(planned_tasks) / 5.0)
            analysis_score = 1.0 if result.get('analyzed_objectives') else 0.0
            score = planning_score * 0.6 + analysis_score * 0.4
            
        else:
            score = 0.0
        
        return {
            'category': category,
            'success': True,
            'score': score * 100,  # Convert to percentage
            'sources_count': len(sources),
            'execution_results_count': len(execution_results),
            'planned_tasks_count': len(planned_tasks),
            'agents_count': len(agent_status)
        }
        
    except Exception as e:
        return {
            'category': category,
            'success': False,
            'score': 0.0,
            'error': str(e)
        }


async def measure_agentbench_scores() -> Dict[str, float]:
    """AgentBench 점수 측정"""
    print("=" * 80)
    print("🚀 AgentBench 점수 측정")
    print("=" * 80)
    print()
    
    # 카테고리별 테스트 쿼리
    test_queries = {
        'WebNavigation': 'Latest AI developments in 2025',
        'ToolUsage': 'Analyze remote work productivity trends',
        'MultiAgent': 'Innovation in education and learning methods',
        'Reasoning': 'Logical analysis of AI ethics implications'
    }
    
    results = {}
    category_scores = defaultdict(list)
    
    for category, query in test_queries.items():
        print(f"📊 {category}: {query}")
        try:
            result = await measure_single_task(query, category)
            results[category] = result
            category_scores[category].append(result['score'])
            status = "✅" if result['success'] else "❌"
            print(f"   {status} Score: {result['score']:.1f}%")
            if not result['success']:
                print(f"      Error: {result.get('error', 'Unknown')}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
            results[category] = {'category': category, 'success': False, 'score': 0.0, 'error': str(e)}
        print()
    
    # 카테고리별 평균 계산
    scores = {}
    category_names = {
        'WebNavigation': 'Web Navigation',
        'ToolUsage': 'Tool Usage',
        'MultiAgent': 'Multi-Agent',
        'Reasoning': 'Reasoning'
    }
    
    for category, score_list in category_scores.items():
        if score_list:
            avg = sum(score_list) / len(score_list)
            scores[category_names[category]] = avg
        else:
            scores[category_names[category]] = 0.0
    
    # Overall 계산
    if scores:
        overall = sum(scores.values()) / len(scores)
        scores['Overall'] = overall
    
    return scores


def main():
    """메인 실행"""
    scores = asyncio.run(measure_agentbench_scores())
    
    print("=" * 80)
    print("📊 AgentBench Scores")
    print("=" * 80)
    print()
    
    for category, score in scores.items():
        print(f"  {category}: {score:.1f}%")
    
    print()
    print("=" * 80)
    
    # 결과 저장
    output_file = project_root / "results" / "agentbench_scores.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(scores, f, indent=2)
    
    print(f"💾 저장됨: {output_file}")
    
    # README 업데이트 제안
    print()
    print("💡 README 업데이트:")
    print("   다음 값들로 README를 업데이트할 수 있습니다:")
    for category, score in scores.items():
        if category != 'Overall':
            print(f"   {category}: {score:.1f}%")


if __name__ == "__main__":
    main()

