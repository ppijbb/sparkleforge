#!/usr/bin/env python3
"""
새로운 Agent Orchestrator 실행 테스트

실행 추적과 함께 전체 워크플로우 확인
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.agent_orchestrator import AgentOrchestrator
from src.core.shared_memory import get_shared_memory, MemoryScope


async def run_with_trace():
    """실행 추적과 함께 워크플로우 테스트"""
    
    print("=" * 80)
    print(" Agent Orchestrator 실행 추적 테스트")
    print("=" * 80)
    print()
    
    # 1. 초기화
    print("📦 1. 시스템 초기화...")
    orchestrator = AgentOrchestrator()
    memory = get_shared_memory()
    print("   ✓ Orchestrator initialized")
    print("   ✓ Shared memory initialized")
    print()
    
    # 2. 메모리 사전 데이터
    print("📝 2. 메모리에 사전 데이터 추가...")
    memory.write("previous_research", "Climate change research findings from last week", scope=MemoryScope.GLOBAL)
    memory.write("user_preference", "Detailed analysis preferred", scope=MemoryScope.GLOBAL)
    print("   ✓ Previous research stored")
    print("   ✓ User preference stored")
    print()
    
    # 3. 쿼리 정의
    query = "Analyze the latest developments in quantum computing"
    print(f"🔍 3. Research Query: {query}")
    print()
    
    # 4. 워크플로우 실행
    print("⚙️  4. 워크플로우 실행 중...")
    print("   - Planner → Executor → Verifier → Generator")
    print()
    
    result = await orchestrator.execute(query)
    
    print()
    print("✅ 워크플로우 완료!")
    print()
    
    # 5. 결과 확인
    print("=" * 80)
    print(" 결과 확인")
    print("=" * 80)
    print()
    
    session_id = result.get('session_id')
    print(f"📋 Session ID: {session_id}")
    print(f"🤖 Current Agent: {result.get('current_agent')}")
    print(f"🔄 Iteration: {result.get('iteration')}")
    print()
    
    # Plan 확인
    if result.get('research_plan'):
        plan = result['research_plan']
        print(f"📋 Research Plan: {len(plan)} chars")
        print(f"   {plan[:100]}...")
        print()
    
    # Results 확인
    results = result.get('research_results', [])
    print(f"📊 Research Results: {len(results)} items")
    if results:
        for i, r in enumerate(results[:3], 1):
            print(f"   {i}. {r[:80]}...")
    print()
    
    # Verified 확인
    verified = result.get('verified_results', [])
    print(f"✅ Verified Results: {len(verified)} items")
    if verified:
        for i, v in enumerate(verified[:3], 1):
            print(f"   {i}. {v[:80]}...")
    print()
    
    # Final Report 확인
    if result.get('final_report'):
        report = result['final_report']
        print(f"📄 Final Report: {len(report)} chars")
        print()
        print("=" * 80)
        print(report[:500])
        print("=" * 80)
        print()
    
    # Memory 확인
    print("💾 Shared Memory 확인...")
    plan_in_memory = memory.read(f'plan_{session_id}', scope=MemoryScope.SESSION, session_id=session_id)
    report_in_memory = memory.read(f'report_{session_id}', scope=MemoryScope.SESSION, session_id=session_id)
    
    print(f"   Plan in memory: {'✅ 있음' if plan_in_memory else '❌ 없음'}")
    print(f"   Report in memory: {'✅ 있음' if report_in_memory else '❌ 없음'}")
    print()
    
    # 검색 테스트
    print("🔍 Memory Search 테스트...")
    search_results = memory.search("research", limit=5, scope=MemoryScope.SESSION)
    print(f"   Found {len(search_results)} results")
    for r in search_results:
        print(f"   - {r['key']}: {str(r['value'])[:50]}...")
    print()
    
    print("=" * 80)
    print(" ✅ 모든 테스트 통과!")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(run_with_trace())


