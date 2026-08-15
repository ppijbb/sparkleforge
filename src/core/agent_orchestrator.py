"""Thin Wrapper for Agent Orchestrator.

기존 8,500라인의 거대한 AgentOrchestrator를 대체하는 얇은 래퍼입니다.
실제 실행과 로직은 2026 기반 AgentHarness와 독립된 서브 에이전트들이 담당합니다.
"""

import logging
import os
from typing import Any, Dict, List, TypedDict
from src.core.isomorphism_extractor import CrossDomainIsomorphismExtractor
# Removed global AgentHarness import for optimization


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
        heat_seconds: float | None = None,
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

        # Wire the Gemini prompt-cache handle into the harness call path so the
        # cache plumbing defined above is actually read from and written to.
        # See issue #778: previously the handle was built but never connected.
        cached_handle = self.get_cached_handle(session_id)
        if cached_handle is not None:
            kwargs.setdefault("gemini_cached_handle", cached_handle)
        
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

        # classify -> TaskRouter(LLM) -> planner/single_agent 그래프를 기본 실행 경로로 사용해
        # 실제로 라우팅 판단이 이루어지게 한다. heat_seconds(--heat 시간 예산, issue #585)는
        # wrap-up 리포트 기능이 autonomous 루프에만 구현되어 있으므로 그 경우만 예외로 둔다.
        # coworker 세션(identity="coder")은 항상 autonomous Hermes 루프를 써야 한다 --
        # 그렇지 않으면 로컬 코딩 요청까지 전부 research LangGraph로 강제 라우팅된다.
        mode = "autonomous" if heat_seconds or identity == "coder" else "research"
        harness_result = await self.harness.execute(
            session_id=session_id,
            request=request,
            max_iterations=max_iterations,
            mode=mode,
            identity=identity,
            heat_seconds=heat_seconds,
            custom_state=custom_state,
        )

        # Persist coworker sessions so they can be resumed/approved/denied later
        if identity == "coder" and session_id:
            try:
                from src.core.session_manager import get_session_manager
                get_session_manager().save_session(
                    session_id, agent_state=harness_result, metadata={"tags": ["coworker"]}
                )
            except Exception:
                logger.debug("Session persistence skipped for coworker session %s", session_id, exc_info=True)

        # Persist any Gemini prompt-cache handle produced by the harness run
        # back into the per-session store so subsequent turns can reuse it.
        new_handle = (
            harness_result.get("metadata", {}).get("gemini_cached_handle")
            if isinstance(harness_result.get("metadata"), dict)
            else None
        )
        if new_handle and self.gemini_cache is not None:
            self.set_cached_handle(session_id, new_handle)

        # main.py 호환을 위한 필드 추가
        final_report = harness_result.get("results", "")
        return {
            "success": harness_result.get("success", False),
            "plan": harness_result.get("plan", ""),
            "tasks": harness_result.get("tasks", []),
            "results": final_report,
            "final_report": final_report,
            "metadata": harness_result.get("metadata", {}),
            "session_id": session_id,
            "research_failed": not harness_result.get("success", False),
            "error": harness_result.get("error"),
        }
