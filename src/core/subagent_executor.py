"""Context-Isolated Sub-Agent Executor.

서브에이전트는 부모 에이전트와 독립된 컨텍스트(Fresh Context Window) 위에서 실행되며,
결과 데이터 오염을 방지하기 위해 Handle 방식으로 결과를 전달합니다.
"""

import asyncio
import logging
import time
import uuid
from typing import Any, Dict

from src.core.harness_state import TaskState
from src.core.sub_agent_manager import (
    SubAgentConfig,
    SubAgentPerformanceStore,
    get_sub_agent_manager,
)
from src.core.tool_governor import get_tool_governor

logger = logging.getLogger(__name__)


class SubAgentExecutor:
    """격리된 생태계에서 서브에이전트를 실행하는 전용 실행 엔진"""

    def __init__(self):
        self.tool_governor = get_tool_governor()
        self.sam = get_sub_agent_manager()
        self.performance_store = SubAgentPerformanceStore()
        # 간단한 메모리 내 Artifact Store (실제로는 파일이나 DB를 참조해야 함)
        self.artifact_store: Dict[str, Any] = {}

    async def execute_isolated(
        self, task: TaskState, agent_config: SubAgentConfig, timeout: int = 300
    ) -> TaskState:
        """독립된 컨텍스트 공간에서 서브에이전트 실행을 위임합니다."""
        # Capture and propagate the current TrustContext
        from src.core.trust_gate import (
            get_current_trust_context,
            set_current_trust_context,
        )
        try:
            parent_trust = get_current_trust_context()
            set_current_trust_context(parent_trust)
        except Exception as trust_err:
            logger.warning(f"Failed to propagate TrustContext to sub-agent: {trust_err}")

        task_id = task.get("task_id", str(uuid.uuid4()))
        agent_name = agent_config.name
        logger.info(
            f"🚀 [SubAgentExecutor] Starting isolated execution of task {task_id} on {agent_name}..."
        )

        start_time = time.time()

        # 1. Fresh Context Window (부모의 거대한 prompt / message 없이 깨끗하게 시작)
        isolated_messages = [
            {
                "role": "system",
                "content": f"You are a specialized agent: {agent_config.role.value}. Specialization: {agent_config.specialization_area}",
            },
            {
                "role": "user",
                "content": f"Please execute the following task:\n\n{task.get('description')}",
            },
        ]

        # 향후 LLM 통합 시 이 부분에 LLM 호출 모델을 바인딩 됨.
        # model = get_llm_for_subagent(agent_config)
        # response = await model.ainvoke(isolated_messages)

        # 타임아웃 감시용 비동기 Task 래핑
        try:
            result = await asyncio.wait_for(
                self._dummy_agent_loop(task, agent_config, isolated_messages), timeout=timeout
            )

            execution_time = time.time() - start_time

            # 2. 결과물 격리 보관 (Result Handle 반환 패턴)
            artifact_id = f"artifact_{task_id}_{uuid.uuid4().hex[:8]}"
            self.artifact_store[artifact_id] = result["data"]

            # 3. TaskState 갱신 (핸들만 기록)
            task["status"] = "completed"
            task["assigned_agent"] = agent_name
            task["result"] = {
                "artifact_id": artifact_id,
                "summary": result.get("summary", "작업 완료됨"),
            }
            task["execution_time"] = execution_time

            # 4. Performance Tracking
            self._record_performance(agent_name, agent_config.role.value, True, execution_time)

            logger.info(f"✅ [SubAgentExecutor] Task {task_id} completed. Artifact: {artifact_id}")

        except TimeoutError:
            logger.error(f"❌ [SubAgentExecutor] Task {task_id} timed out after {timeout}s.")
            execution_time = time.time() - start_time
            task["status"] = "failed"
            task["error"] = "Execution Timeout"
            task["execution_time"] = execution_time
            self._record_performance(agent_name, agent_config.role.value, False, execution_time)

        except Exception as e:
            logger.error(f"❌ [SubAgentExecutor] Task {task_id} failed: {e}")
            execution_time = time.time() - start_time
            task["status"] = "failed"
            task["error"] = str(e)
            task["execution_time"] = execution_time
            self._record_performance(agent_name, agent_config.role.value, False, execution_time)

        return task

    async def _dummy_agent_loop(
        self, task: TaskState, agent_config: SubAgentConfig, messages: list
    ) -> Dict[str, Any]:
        """서브에이전트의 내부 실행 루프"""
        # ToolGovernor 경유하여 도구 호출 예시
        if "search" in agent_config.capabilities:
            query = f"{task.get('description', '')[:50]}"
            await self.tool_governor.execute_tool("search_web", {"query": query})

        return {
            "summary": f"{agent_config.name} successfully handled the task.",
            "data": {
                "status": "success",
                "agent_details": agent_config.name,
                "task_desc": task.get("description"),
            },
        }

    def _record_performance(self, agent_name: str, role: str, success: bool, execution_time: float):
        """성과 측정 로직 - 향후 고성과 에이전트 재활용(Dynamic SubAgent Factory)에 활용됨"""
        try:
            self.performance_store.record_performance(
                agent_name=agent_name,
                role=role,
                success=success,
                execution_time=execution_time,
                task_complexity=1.0,  # 기본값
            )
        except Exception as e:
            logger.error(f"Failed to record performance for {agent_name}: {e}")


# 전역 인스턴스
_executor = None


def get_subagent_executor() -> SubAgentExecutor:
    global _executor
    if _executor is None:
        _executor = SubAgentExecutor()
    return _executor
