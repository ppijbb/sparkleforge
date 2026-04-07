"""Thin Wrapper for Agent Orchestrator.

기존 8,500라인의 거대한 AgentOrchestrator를 대체하는 얇은 래퍼입니다.
실제 실행과 로직은 2026 기반 AgentHarness와 독립된 서브 에이전트들이 담당합니다.
"""

import logging
from typing import Dict, Any, List

from src.core.agent_harness import AgentHarness
from src.core.task_router import RoutePath

logger = logging.getLogger(__name__)

class AgentOrchestrator:
    """Agent Harness 기반의 경량화된 Orchestrator Wrapper"""
    
    def __init__(self):
        self.harness = AgentHarness()
        logger.info("AgentOrchestrator initialized with AgentHarness")

    async def execute(self, request: str, session_id: str = "default_session", max_iterations: int = 10) -> Dict[str, Any]:
        """하네스를 기동하여 요청을 처리합니다."""
        logger.info(f"AgentOrchestrator delegating request to AgentHarness (session: {session_id})")
        
        # Harness 실행
        return await self.harness.execute(
            session_id=session_id, 
            request=request, 
            max_iterations=max_iterations
        )

def agent_workflow_result_to_public_dict(result: Dict[str, Any]) -> Dict[str, Any]:
    """이전 버전의 API 호환성을 위한 포맷터"""
    return {
        "plan": result.get("plan", ""),
        "tasks": result.get("tasks", []),
        "results": result.get("results", ""),
        "success": result.get("success", False)
    }
