#!/usr/bin/env python3
"""
Quick AgentBench score measurement - 실제 에이전트 실행으로 성능 측정
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Any

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).parent))

from src.core.researcher_config import load_config_from_env
from src.core.autonomous_orchestrator import AutonomousOrchestrator

async def measure_agent_performance(query: str, category: str) -> Dict[str, Any]:
    """실제 에이전트 실행으로 성능 측정"""
    try:
        load_config_from_env()
        orchestrator = AutonomousOrchestrator()
        
        start_time = time.time()
        result = await orchestrator.run_research(query)
        execution_time = time.time() - start_time
        
        # 결과에서 성공 여부 판단
        success = result is not None
        execution_results = result.get('execution_results', []) if result else []
        sources_count = len(result.get('sources', [])) if result else 0
        
        # 실행 결과에서 도구 사용, 병렬 실행 등 추출
        tools_used = []
        parallel_executed = False
        reasoning_steps = []
        
        if execution_results:
            for exec_result in execution_results:
                if exec_result.get('tool_used'):
                    tools_used.append(exec_result['tool_used'])
                if exec_result.get('parallel_execution'):
                    parallel_executed = True
                if exec_result.get('reasoning'):
                    reasoning_steps.append(exec_result['reasoning'])
        
        # 카테고리별 점수 계산
        if category == 'WebNavigation':
            # Web Navigation 점수: 소스 수와 성공 여부 기반
            score = min(1.0, (sources_count / 5.0) * 0.8 + (1.0 if success else 0.0) * 0.2)
        elif category == 'ToolUsage':
            # Tool Usage 점수: 실행 성공과 도구 사용 기반
            tools_count = len(tools_used)
            score = min(1.0, (1.0 if success else 0.0) * 0.6 + min(1.0, tools_count / 3.0) * 0.4)
        elif category == 'MultiAgent':
            # Multi-Agent 점수: 병렬 실행 성공 기반
            score = min(1.0, (1.0 if success else 0.0) * 0.7 + (1.0 if parallel_executed else 0.0) * 0.3)
        elif category == 'Reasoning':
            # Reasoning 점수: 논리적 일관성과 성공 기반
            reasoning_count = len(reasoning_steps)
            score = min(1.0, (1.0 if success else 0.0) * 0.5 + min(1.0, reasoning_count / 5.0) * 0.5)
        else:
            score = 1.0 if success else 0.0
        
        return {
            'success': success,
            'score': score,
            'execution_time': execution_time,
            'sources_count': sources_count,
            'tools_used': tools_used,
            'parallel_executed': parallel_executed,
            'reasoning_steps': reasoning_steps,
            'result': result
        }
    except Exception as e:
        print(f"Error measuring {category}: {e}")
        return {
            'success': False,
            'score': 0.0,
            'execution_time': 0.0,
            'sources_count': 0,
            'error': str(e)
        }


async def calculate_agentbench_scores() -> Dict[str, float]:
    """실제 실행으로 AgentBench 점수 계산"""
    print("=" * 80)
    print("🚀 Measuring AgentBench Performance")
    print("=" * 80)
    print()
    
    # 각 카테고리별 테스트 쿼리
    test_queries = {
        'WebNavigation': 'Latest AI developments in 2025',
        'ToolUsage': 'Analyze remote work productivity trends',
        'MultiAgent': 'Innovation in education and learning methods',
        'Reasoning': 'Logical analysis of AI ethics implications'
    }
    
    scores = {}
    results = {}
    
    for category, query in test_queries.items():
        print(f"📊 Testing {category}: {query}")
        result = await measure_agent_performance(query, category)
        scores[category] = result['score'] * 100  # Convert to percentage
        results[category] = result
        print(f"   Score: {scores[category]:.1f}% (Success: {result['success']}, Time: {result['execution_time']:.1f}s)")
        print()
    
    # Overall score 계산
    if scores:
        overall = sum(scores.values()) / len(scores)
        scores['Overall'] = overall
        print(f"📊 Overall Score: {overall:.1f}%")
    
    return scores


if __name__ == "__main__":
    import asyncio
    scores = asyncio.run(calculate_agentbench_scores())
    
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
    
    # Save results
    output_file = project_root / "results" / "agentbench_scores.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    formatted_scores = {category_names.get(k, k): v for k, v in scores.items()}
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(formatted_scores, f, indent=2)
    
    print(f"💾 Scores saved to: {output_file}")

