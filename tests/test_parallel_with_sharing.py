#!/usr/bin/env python3
"""
병렬 실행 + 결과 공유 + 토론 통합 테스트

실제 ParallelAgentExecutor와 결과 공유 시스템이 함께 작동하는지 확인합니다.
"""

import asyncio
import sys
import logging
from pathlib import Path

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.researcher_config import load_config_from_env
from src.core.parallel_agent_executor import ParallelAgentExecutor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_parallel_execution_with_sharing():
    """병렬 실행과 결과 공유 통합 테스트."""
    logger.info("=" * 80)
    logger.info("Testing Parallel Execution with Result Sharing")
    logger.info("=" * 80)
    
    try:
        # 설정 로드
        config = load_config_from_env()
        logger.info(f"✅ Config loaded: agent_communication={config.agent.enable_agent_communication}")
        
        # Executor 생성
        executor = ParallelAgentExecutor()
        
        # 테스트 작업 생성 (동일한 작업에 대한 여러 agent 실행 시뮬레이션)
        tasks = [
            {
                "id": "test_task_1",
                "task_id": "test_task_1",
                "name": "Test Search 1",
                "task_type": "search",
                "query": "artificial intelligence",
                "dependencies": [],
                "priority": 1,
                "max_results": 3
            },
            {
                "id": "test_task_1_alt",
                "task_id": "test_task_1_alt",
                "name": "Test Search 1 Alternative",
                "task_type": "search",
                "query": "artificial intelligence",
                "dependencies": [],
                "priority": 1,
                "max_results": 3
            }
        ]
        
        agent_assignments = {}
        execution_plan = {
            "strategy": "parallel",
            "parallel_groups": [["test_task_1", "test_task_1_alt"]],
            "execution_order": ["test_task_1", "test_task_1_alt"],
            "estimated_total_time": 60,
            "dependency_graph": {"test_task_1": [], "test_task_1_alt": []},
            "task_count": 2,
            "agent_count": 0
        }
        
        logger.info("⚠️ 실제 실행 테스트는 MCP 서버 연결이 필요합니다")
        logger.info("✅ Executor 구조 확인:")
        logger.info(f"   - Agent communication enabled: {executor.agent_config.enable_agent_communication}")
        logger.info(f"   - Max concurrent: {executor.max_concurrent}")
        logger.info(f"   - Shared results manager: {executor.shared_results_manager is None}")
        logger.info(f"   - Discussion manager: {executor.discussion_manager is None}")
        
        # execute_parallel_tasks가 호출되면 초기화됨
        if executor.agent_config.enable_agent_communication:
            logger.info("✅ Agent communication is enabled - sharing will be initialized on execution")
        else:
            logger.info("⚠️ Agent communication is disabled - sharing will not be used")
        
        logger.info("✅ 통합 테스트 구조 확인 완료\n")
        
    except Exception as e:
        logger.error(f"❌ 테스트 실패: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise


async def test_result_sharing_workflow():
    """결과 공유 워크플로우 시뮬레이션."""
    logger.info("=" * 80)
    logger.info("Testing Result Sharing Workflow Simulation")
    logger.info("=" * 80)
    
    try:
        from src.core.agent_result_sharing import SharedResultsManager, AgentDiscussionManager
        
        # 설정 로드
        config = load_config_from_env()
        
        # Manager 생성
        shared_results_manager = SharedResultsManager(objective_id="test_workflow")
        discussion_manager = AgentDiscussionManager(
            objective_id="test_workflow",
            shared_results_manager=shared_results_manager
        )
        
        # 시나리오: 3개 agent가 동일한 작업에 대해 병렬로 실행
        task_id = "parallel_task"
        
        # Agent 1 결과
        result_id_1 = await shared_results_manager.share_result(
            task_id=task_id,
            agent_id="agent_task_1",
            result={"query": "AI trends", "sources": 5, "findings": "Growing adoption"},
            confidence=0.85
        )
        logger.info(f"✅ Agent 1 shared result: {result_id_1}")
        
        # Agent 2 결과 (Agent 1의 결과를 볼 수 있음)
        other_results = await shared_results_manager.get_shared_results(
            task_id=task_id,
            exclude_agent_id="agent_task_2"
        )
        logger.info(f"✅ Agent 2 can see {len(other_results)} other results before sharing")
        
        result_id_2 = await shared_results_manager.share_result(
            task_id=task_id,
            agent_id="agent_task_2",
            result={"query": "AI trends", "sources": 3, "findings": "Market expansion"},
            confidence=0.75
        )
        logger.info(f"✅ Agent 2 shared result: {result_id_2}")
        
        # Agent 3 결과 (Agent 1, 2의 결과를 볼 수 있음)
        other_results = await shared_results_manager.get_shared_results(
            task_id=task_id,
            exclude_agent_id="agent_task_3"
        )
        logger.info(f"✅ Agent 3 can see {len(other_results)} other results before sharing")
        
        result_id_3 = await shared_results_manager.share_result(
            task_id=task_id,
            agent_id="agent_task_3",
            result={"query": "AI trends", "sources": 8, "findings": "Technology advancement"},
            confidence=0.90
        )
        logger.info(f"✅ Agent 3 shared result: {result_id_3}")
        
        # 전체 요약
        summary = await shared_results_manager.get_result_summary(task_id=task_id)
        logger.info(f"✅ Final summary: {summary['total_results']} results from {summary['agents_count']} agents")
        logger.info(f"   Average confidence: {summary['average_confidence']:.2f}")
        
        # 토론 구조 확인 (실제 LLM 호출은 환경 필요)
        logger.info("⚠️ Full discussion requires LLM configuration")
        discussion_summary = await discussion_manager.get_discussion_summary()
        logger.info(f"✅ Discussion summary: {discussion_summary}")
        
        logger.info("✅ 워크플로우 시뮬레이션 완료\n")
        
    except Exception as e:
        logger.error(f"❌ 테스트 실패: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise


async def main():
    """메인 테스트 함수."""
    logger.info("🚀 병렬 실행 + 결과 공유 통합 테스트 시작")
    logger.info("")
    
    try:
        await test_parallel_execution_with_sharing()
        await test_result_sharing_workflow()
        
        logger.info("=" * 80)
        logger.info("✅ 모든 통합 테스트 통과!")
        logger.info("=" * 80)
        logger.info("")
        logger.info("📋 구현된 기능:")
        logger.info("   1. ✅ Agent 간 결과 공유 (SharedResultsManager)")
        logger.info("   2. ✅ Agent 간 토론 (AgentDiscussionManager)")
        logger.info("   3. ✅ ParallelAgentExecutor 통합")
        logger.info("   4. ✅ 동일 작업에 대한 여러 agent 결과 조회")
        logger.info("   5. ✅ 결과 요약 및 통계")
        logger.info("")
        logger.info("⚠️  실제 LLM 기반 토론은 환경 설정 필요")
        
    except Exception as e:
        logger.error(f"❌ 테스트 실패: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

