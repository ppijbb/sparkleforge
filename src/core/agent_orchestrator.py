"""Thin Wrapper for Agent Orchestrator.

기존 8,500라인의 거대한 AgentOrchestrator를 대체하는 얇은 래퍼입니다.
실제 실행과 로직은 2026 기반 AgentHarness와 독립된 서브 에이전트들이 담당합니다.

Delegation depth is tracked exclusively via ``context`` to keep nested and
concurrent delegations independent (Issue #516).
"""

import logging
import os
from typing import Any, Dict, List, TypedDict

# Removed global AgentHarness import for optimization
from pathlib import Path
from typing import Optional

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
_DELEGATION_ADAPTERS = {
    "codebase_agent": "codebase_agent",
    "document_organizer_agent": "document_organizer_agent",
}


class AgentOrchestrator:
    """Agent Harness 기반의 경량화된 Orchestrator Wrapper"""

    def __init__(self, config=None):
        from src.core.agent_harness import AgentHarness
        self.harness = AgentHarness()
        self.config = config
        self.recursion_limit = getattr(config, "recursion_limit", 20000)
        self.gemini_cache = self._init_gemini_cache()
        self.federation_enabled = getattr(config, "federation_enabled", False)
        logger.info("AgentOrchestrator initialized with AgentHarness")

    def _init_gemini_cache(self):
        """Initialize Gemini prompt caching manager (Issue #459).

        Returns a lightweight cache handle store keyed by session_id. Falls back
        to None when caching is disabled or the google-genai SDK is unavailable.
        """
        try:
            from src.core.researcher_config import GeminiCacheConfig
        except Exception:
            return None

        cache_cfg = GeminiCacheConfig()
        try:
            enabled_env = os.getenv("GEMINI_CACHE_ENABLED")
            if enabled_env is not None:
                cache_cfg.enabled = enabled_env.lower() in ("true", "1", "yes", "on")
            ttl_env = os.getenv("GEMINI_CACHE_TTL_SECONDS")
            if ttl_env:
                cache_cfg.ttl_seconds = int(ttl_env)
        except Exception:
            pass

        if not cache_cfg.enabled:
            return None

        try:
            from google import genai  # noqa: F401
        except Exception:
            logger.info("[GeminiCache] google-genai SDK unavailable; caching disabled")
            return None

        return {"config": cache_cfg, "handles": {}}

    def _is_gemini_model(self, model: str | None) -> bool:
        """Check whether a model name targets a Gemini-family model."""
        if not model:
            return False
        if self.gemini_cache is None:
            return False
        prefixes = self.gemini_cache["config"].gemini_model_prefixes
        return any(model.lower().startswith(p) for p in prefixes)

    def get_gemini_cache_config(self):
        """Expose the active GeminiCacheConfig (or None) for downstream callers."""
        if self.gemini_cache is None:
            return None
        return self.gemini_cache["config"]

    def get_cached_handle(self, session_id: str) -> str | None:
        """Return the cached Gemini context handle for a session, if any."""
        if self.gemini_cache is None:
            return None
        return self.gemini_cache["handles"].get(session_id)

    def set_cached_handle(self, session_id: str, handle: str) -> None:
        """Store a cached Gemini context handle for reuse across turns."""
        if self.gemini_cache is None:
            return
        self.gemini_cache["handles"][session_id] = handle

    def clear_cached_handle(self, session_id: str) -> None:
        """Drop a cached handle (e.g. on expiry or fallback)."""
        if self.gemini_cache is None:
            return
        self.gemini_cache["handles"].pop(session_id, None)

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
        
        # Federation check
        if self.federation_enabled and kwargs.get("federated"):
            from src.core.federation.protocol import FederationClient
            client = FederationClient()
            sub_tasks = await client.distribute_tasks(request)
            if sub_tasks:
                logger.info(f"Federated {len(sub_tasks)} sub-tasks to remote nodes.")
                kwargs["sub_tasks"] = sub_tasks

        logger.info(
            f"AgentOrchestrator delegating request to AgentHarness (session: {session_id}, "
            f"gemini_cache={'on' if self.gemini_cache else 'off'})"
        )

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

    MAX_DELEGATION_DEPTH = 3

    async def delegate_to_agent(
        self,
        agent_name: str,
        context: Dict[str, Any],
        state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Delegate a task to a named agent.

        The delegation depth is read exclusively from ``context`` so that
        nested delegations observe the incremented value even when the parent
        restores its own state. This avoids races on shared ``state`` when
        delegations are dispatched concurrently via ``asyncio.gather``.
        """
        depth = int(context.get("delegation_depth", 0))
        if depth >= self.MAX_DELEGATION_DEPTH:
            from src.core.exceptions import DelegationDepthExceeded

            raise DelegationDepthExceeded(
                f"Delegation depth exceeded for agent '{agent_name}' at depth {depth}"
            )

        child_context = {**context, "delegation_depth": depth + 1}

        adapter = self._get_delegation_adapter(agent_name)
        return await adapter.execute(child_context, state)

    def _get_delegation_adapter(self, agent_name: str):
        """Return the adapter for the named agent."""
        if agent_name == "codebase_agent":
            return _CodebaseAgentAdapter(self.harness)
        if agent_name == "document_organizer_agent":
            return _DocumentOrganizerAgentAdapter(self.harness)
        raise ValueError(f"Unknown agent: {agent_name}")

    async def _delegate_codebase_agent(self, context: Dict[str, Any]):
        path = context.get("path")
        if not path:
            raise ValueError("'path' is required for codebase delegation")
        path = Path(path)
        return await self.delegate_to_agent("codebase_agent", context)

    async def _delegate_document_organizer_agent(self, context: Dict[str, Any]):
        path = context.get("path")
        if not path:
            raise ValueError("'path' is required for document organizer delegation")
        path = Path(path)
        return await self.delegate_to_agent("document_organizer_agent", context)


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


class _CodebaseAgentAdapter:
    """Adapter for codebase_agent delegation."""

    def __init__(self, harness):
        self.harness = harness

    async def execute(self, context, state):
        return await self.harness.execute(
            session_id=context.get("session_id", "delegation"),
            request=context.get("request", ""),
            identity="coder",
        )


class _DocumentOrganizerAgentAdapter:
    """Adapter for document_organizer_agent delegation."""

    def __init__(self, harness):
        self.harness = harness

    async def execute(self, context, state):
        return await self.harness.execute(
            session_id=context.get("session_id", "delegation"),
            request=context.get("request", ""),
            identity="researcher",
        )
