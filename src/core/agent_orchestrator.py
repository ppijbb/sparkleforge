"""Thin Wrapper for Agent Orchestrator.

기존 8,500라인의 거대한 AgentOrchestrator를 대체하는 얇은 래퍼입니다.
실제 실행과 로직은 2026 기반 AgentHarness와 독립된 서브 에이전트들이 담당합니다.
"""

import logging
from typing import Any, Dict, List, TypedDict

from src.core.agent_harness import AgentHarness


class AgentState(TypedDict, total=False):
    """Agent workflow state shared across orchestration steps."""

    request: str
    session_id: str
    plan: str
    tasks: List[Dict[str, Any]]
    results: str
    final_report: str
    success: bool
    error: str | None


logger = logging.getLogger(__name__)


class AgentOrchestrator:
    """Agent Harness 기반의 경량화된 Orchestrator Wrapper"""

    def __init__(self, config=None):
        self.harness = AgentHarness()
        self.config = config
        self.recursion_limit = getattr(config, "recursion_limit", 20000)
        logger.info("AgentOrchestrator initialized with AgentHarness")

    async def execute(
        self, request: str, session_id: str = "default_session", max_iterations: int = 10, **kwargs
    ) -> Dict[str, Any]:
        """하네스를 기동하여 요청을 처리합니다."""
        logger.info(f"AgentOrchestrator delegating request to AgentHarness (session: {session_id})")

        # Harness 실행
        harness_result = await self.harness.execute(
            session_id=session_id, request=request, max_iterations=max_iterations
        )

        # main.py 호환을 위한 필드 추가
        return {
            "success": harness_result.get("success", False),
            "plan": harness_result.get("plan", ""),
            "tasks": harness_result.get("tasks", []),
            "results": harness_result.get("results", ""),
            "final_report": harness_result.get("results", ""),  # results를 final_report로 매핑
            "session_id": session_id,
            "research_failed": not harness_result.get("success", False),
            "error": harness_result.get("error"),
        }


def agent_workflow_result_to_public_dict(
    result: Dict[str, Any], context: Dict[str, Any] | None = None
) -> Dict[str, Any]:
    """이전 버전의 API 호환성을 위한 포맷터"""
    return {
        "plan": result.get("plan", ""),
        "tasks": result.get("tasks", []),
        "results": result.get("results", ""),
        "success": result.get("success", False),
    }
