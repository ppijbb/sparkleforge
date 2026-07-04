"""Thin Wrapper for Agent Orchestrator.

기존 8,500라인의 거대한 AgentOrchestrator를 대체하는 얇은 래퍼입니다.
실제 실행과 로직은 2026 기반 AgentHarness와 독립된 서브 에이전트들이 담당합니다.
"""

import logging
import os
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
_orchestrator: "AgentOrchestrator | None" = None


class AgentOrchestrator:
    """Agent Harness 기반의 경량화된 Orchestrator Wrapper"""

    def __init__(self, config=None):
        self.harness = AgentHarness()
        self.config = config
        self.recursion_limit = getattr(config, "recursion_limit", 20000)
        logger.info("AgentOrchestrator initialized with AgentHarness")

    async def execute(
        self,
        request: str | None = None,
        session_id: str | None = "default_session",
        max_iterations: int | None = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """하네스를 기동하여 요청을 처리합니다."""
        if request is None:
            request = kwargs.pop("user_query", None)
        if request is None:
            raise TypeError("AgentOrchestrator.execute() requires 'request' or 'user_query'")
        if session_id is None:
            session_id = "default_session"
        if max_iterations is None:
            max_iterations = int(os.getenv("SPARKLEFORGE_MAX_ITERATIONS", "30"))

        # coworker 모드는 로컬 저장소를 다루는 coder 페르소나로 실행
        custom_state = kwargs.get("custom_state") or {}
        identity = "coder" if custom_state.get("mode") == "coworker" else "researcher"

        logger.info(f"AgentOrchestrator delegating request to AgentHarness (session: {session_id})")

        # Harness 실행
        harness_result = await self.harness.execute(
            session_id=session_id,
            request=request,
            max_iterations=max_iterations,
            identity=identity,
        )

        # main.py 호환을 위한 필드 추가
        final_report = harness_result.get("results", "")
        return {
            "success": harness_result.get("success", False),
            "plan": harness_result.get("plan", ""),
            "tasks": harness_result.get("tasks", []),
            "results": final_report,
            "final_report": final_report,  # results를 final_report로 매핑
            "content": final_report,
            "detailed_results": {
                "plan": harness_result.get("plan", ""),
                "tasks": harness_result.get("tasks", []),
                "results": final_report,
                "final_report": final_report,
                "success": harness_result.get("success", False),
                "error": harness_result.get("error"),
            },
            "session_id": session_id,
            "research_failed": not harness_result.get("success", False),
            "error": harness_result.get("error"),
        }


def get_orchestrator(config=None) -> AgentOrchestrator:
    """이전 CLI 코드와의 호환성을 위한 lazy singleton accessor."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = AgentOrchestrator(config=config)
    return _orchestrator


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
