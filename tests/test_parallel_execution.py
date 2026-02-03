#!/usr/bin/env python3
"""
병렬 Agent 실행 시스템 테스트

병렬 실행 시스템의 기능을 검증합니다:
- TaskQueue: 작업 큐 및 병렬 그룹 식별
- AgentPool: Agent 풀 관리
- ParallelAgentExecutor: 병렬 실행 관리
"""

import asyncio
import sys
import logging
from pathlib import Path

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.task_queue import TaskQueue
from src.core.agent_pool import AgentPool
from src.core.parallel_agent_executor import ParallelAgentExecutor
from src.core.researcher_config import load_config_from_env

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_task_queue():
    """TaskQueue 테스트."""
    logger.info("=" * 80)
    logger.info("Testing TaskQueue")
    logger.info("=" * 80)
    
    queue = TaskQueue()
    
    # 테스트 작업 생성
    tasks = [
        {
            "id": "task_1",
            "task_id": "task_1",
            "name": "Search for AI trends",
            "task_type": "search",
            "dependencies": [],
            "priority": 1
        },
        {
            "id": "task_2",
            "task_id": "task_2",
            "name": "Search for ML frameworks",
            "task_type": "search",
            "dependencies": [],
            "priority": 1
        },
        {
            "id": "task_3",
            "task_id": "task_3",
            "name": "Analyze results",
            "task_type": "data",
            "dependencies": ["task_1", "task_2"],
            "priority": 2
        }
    ]
    
    # 작업 추가
    queue.add_tasks(tasks)
    
    # 병렬 그룹 확인
    assert len(queue.parallel_groups) > 0, "병렬 그룹이 식별되지 않았습니다"
    logger.info(f"✅ 병렬 그룹 식별: {len(queue.parallel_groups)}개")
    
    # 다음 작업 그룹 가져오기
    next_group = queue.get_next_task_group()
    assert next_group is not None, "다음 작업 그룹을 가져올 수 없습니다"
    assert len(next_group) == 2, f"병렬 그룹 크기가 예상과 다릅니다: {len(next_group)}"
    logger.info(f"✅ 다음 작업 그룹: {next_group}")
    
    # 작업 완료 표시
    for task_id in next_group:
        queue.mark_completed(task_id)
    
    # 진행 상황 확인
    progress = queue.get_progress()
    assert progress['completed_tasks'] == 2, f"완료된 작업 수가 예상과 다릅니다: {progress['completed_tasks']}"
    logger.info(f"✅ 진행 상황: {progress}")
    
    # 다음 그룹 가져오기 (의존성 해결 후)
    next_group = queue.get_next_task_group()
    assert next_group is not None, "의존성 해결 후 다음 그룹을 가져올 수 없습니다"
    assert "task_3" in next_group, "task_3이 다음 그룹에 포함되지 않았습니다"
    logger.info(f"✅ 의존성 해결 후 다음 그룹: {next_group}")
    
    logger.info("✅ TaskQueue 테스트 통과\n")


async def test_agent_pool():
    """AgentPool 테스트."""
    logger.info("=" * 80)
    logger.info("Testing AgentPool")
    logger.info("=" * 80)
    
    pool = AgentPool(max_pool_size=5)
    
    # 간단한 agent 팩토리
    async def create_agent(agent_type: str):
        return {"type": agent_type, "id": f"agent_{agent_type}"}
    
    # Agent 가져오기
    agent1 = await pool.get_agent("researcher", create_agent)
    assert agent1 is not None, "Agent를 가져올 수 없습니다"
    logger.info(f"✅ Agent 생성: {agent1}")
    
    # Agent 반환
    returned = await pool.return_agent("researcher", agent1)
    assert returned, "Agent 반환이 실패했습니다"
    logger.info("✅ Agent 반환 성공")
    
    # 통계 확인
    stats = pool.get_pool_stats()
    assert stats['agent_types']['researcher']['total'] == 1, "Agent 수가 예상과 다릅니다"
    assert stats['agent_types']['researcher']['available'] == 1, "사용 가능한 Agent 수가 예상과 다릅니다"
    logger.info(f"✅ 풀 통계: {stats}")
    
    # 재사용 확인
    agent2 = await pool.get_agent("researcher", create_agent)
    assert agent2 == agent1, "Agent 재사용이 실패했습니다"
    logger.info("✅ Agent 재사용 성공")
    
    logger.info("✅ AgentPool 테스트 통과\n")


async def test_parallel_executor_basic():
    """ParallelAgentExecutor 기본 테스트."""
    logger.info("=" * 80)
    logger.info("Testing ParallelAgentExecutor (Basic)")
    logger.info("=" * 80)
    
    try:
        # 설정 로드
        config = load_config_from_env()
        
        # Executor 생성
        executor = ParallelAgentExecutor()
        
        # 기본 설정 확인
        assert executor.max_concurrent > 0, "max_concurrent가 설정되지 않았습니다"
        assert executor.task_queue is not None, "task_queue가 초기화되지 않았습니다"
        assert executor.agent_pool is not None, "agent_pool이 초기화되지 않았습니다"
        logger.info(f"✅ Executor 초기화 성공: max_concurrent={executor.max_concurrent}")
        
        # 도구 카테고리 확인
        category = executor._get_tool_category_for_task({"task_type": "search"})
        assert category == "search", f"카테고리가 예상과 다릅니다: {category}"
        logger.info(f"✅ 도구 카테고리 식별: {category}")
        
        logger.info("✅ ParallelAgentExecutor 기본 테스트 통과\n")
        
    except Exception as e:
        logger.error(f"❌ 테스트 실패: {e}")
        raise


async def test_integration():
    """통합 테스트: 전체 워크플로우."""
    logger.info("=" * 80)
    logger.info("Testing Integration (Full Workflow)")
    logger.info("=" * 80)
    
    try:
        # 설정 로드
        config = load_config_from_env()
        
        # Executor 생성
        executor = ParallelAgentExecutor()
        
        # 테스트 작업 생성 (의존성 없는 병렬 실행 가능한 작업들)
        tasks = [
            {
                "id": "test_task_1",
                "task_id": "test_task_1",
                "name": "Test Search 1",
                "task_type": "search",
                "query": "artificial intelligence",
                "dependencies": [],
                "priority": 1,
                "max_results": 5
            },
            {
                "id": "test_task_2",
                "task_id": "test_task_2",
                "name": "Test Search 2",
                "task_type": "search",
                "query": "machine learning",
                "dependencies": [],
                "priority": 1,
                "max_results": 5
            }
        ]
        
        agent_assignments = {}
        execution_plan = {
            "strategy": "parallel",
            "parallel_groups": [["test_task_1", "test_task_2"]],
            "execution_order": ["test_task_1", "test_task_2"],
            "estimated_total_time": 60,
            "dependency_graph": {"test_task_1": [], "test_task_2": []},
            "task_count": 2,
            "agent_count": 0
        }
        
        logger.info("⚠️ 실제 실행 테스트는 환경 설정이 필요합니다 (MCP 서버 연결 등)")
        logger.info("✅ 통합 테스트 구조 확인 완료\n")
        
    except Exception as e:
        logger.error(f"❌ 통합 테스트 실패: {e}")
        raise


async def main():
    """메인 테스트 함수."""
    logger.info("🚀 병렬 Agent 실행 시스템 테스트 시작")
    logger.info("")
    
    try:
        # 기본 컴포넌트 테스트
        await test_task_queue()
        await test_agent_pool()
        await test_parallel_executor_basic()
        await test_integration()
        
        logger.info("=" * 80)
        logger.info("✅ 모든 테스트 통과!")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"❌ 테스트 실패: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

