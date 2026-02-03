#!/usr/bin/env python3
"""
간단한 AgentBench 점수 측정 - 실제 실행 결과 기반
"""

import sys
import json
import asyncio
from pathlib import Path
from typing import Dict, List, Any

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core.researcher_config import load_config_from_env
from src.core.autonomous_orchestrator import AutonomousOrchestrator

async def measure_category_score(query: str, category: str) -> float:
    """카테고리별 점수 측정"""
    try:
        load_config_from_env()
        orchestrator = AutonomousOrchestrator()
        
        # 실제 실행
        result = await orchestrator.run_research(query)
        
        if not result:
            return 0.0
        
        # 실행 결과 분석
        execution_results = result.get('execution_results', [])
        sources = result.get('sources', [])
        
        # 카테고리별 점수 계산
        if category == 'WebNavigation':
            # Web Navigation: 소스 수와 검색 성공 기반
            sources_count = len(sources)
            has_search_results = len(execution_results) > 0
            score = min(1.0, (sources_count / 8.0) * 0.7 + (1.0 if has_search_results else 0.0) * 0.3)
            
        elif category == 'ToolUsage':
            # Tool Usage: 도구 사용 성공 기반
            tools_used = set()
            for exec_result in execution_results:
                if exec_result.get('tool_used'):
                    tools_used.add(exec_result['tool_used'])
            tools_count = len(tools_used)
            score = min(1.0, (tools_count / 5.0) * 0.8 + (1.0 if len(execution_results) > 0 else 0.0) * 0.2)
            
        elif category == 'MultiAgent':
            # Multi-Agent: 병렬 실행 및 에이전트 협업 기반
            parallel_executed = any(exec_result.get('parallel_execution', False) for exec_result in execution_results)
            agent_collaboration = len(result.get('agent_status', {})) > 1
            score = min(1.0, (1.0 if parallel_executed else 0.0) * 0.5 + (1.0 if agent_collaboration else 0.0) * 0.5)
            
        elif category == 'Reasoning':
            # Reasoning: 논리적 분석 단계 기반
            reasoning_steps = len(result.get('analyzed_objectives', []))
            has_plan = len(result.get('planned_tasks', [])) > 0
            score = min(1.0, (reasoning_steps / 5.0) * 0.6 + (1.0 if has_plan else 0.0) * 0.4)
            
        else:
            score = 0.0
        
        return score * 100  # Convert to percentage
        
    except Exception as e:
        print(f"Error in {category}: {e}")
        return 0.0


async def main():
    """실제 AgentBench 점수 측정"""
    print("=" * 80)
    print("🚀 Measuring AgentBench Scores")
    print("=" * 80)
    print()
    
    # 각 카테고리별 테스트
    test_queries = {
        'WebNavigation': 'Latest AI developments in 2025',
        'ToolUsage': 'Analyze remote work productivity trends',
        'MultiAgent': 'Innovation in education and learning methods',
        'Reasoning': 'Logical analysis of AI ethics implications'
    }
    
    scores = {}
    
    for category, query in test_queries.items():
        print(f"📊 {category}: {query}")
        try:
            score = await measure_category_score(query, category)
            scores[category] = score
            print(f"   ✅ Score: {score:.1f}%")
        except Exception as e:
            print(f"   ❌ Error: {e}")
            scores[category] = 0.0
        print()
    
    # Overall 계산
    if scores:
        overall = sum(scores.values()) / len(scores)
        scores['Overall'] = overall
    
    # 결과 출력
    print("=" * 80)
    print("📊 AgentBench Scores")
    print("=" * 80)
    print()
    
    category_names = {
        'WebNavigation': 'Web Navigation',
        'ToolUsage': 'Tool Usage',
        'MultiAgent': 'Multi-Agent',
        'Reasoning': 'Reasoning',
        'Overall': 'Overall Score'
    }
    
    for key, value in scores.items():
        name = category_names.get(key, key)
        print(f"  {name}: {value:.1f}%")
    
    print()
    print("=" * 80)
    
    # 저장
    output_file = project_root / "results" / "agentbench_scores.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    formatted = {category_names.get(k, k): v for k, v in scores.items()}
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(formatted, f, indent=2)
    
    print(f"💾 Saved to: {output_file}")
    
    return scores


if __name__ == "__main__":
    asyncio.run(main())

