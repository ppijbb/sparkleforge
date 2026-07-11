"""LangGraph Orchestrator (v3.0 - Modularized Architecture)

Modular architecture refactored from the monolithic 165KB orchestrator.
Delegates core logic to src.core.orchestrator packages.
"""

import logging
import os
from datetime import datetime
from typing import Any, Dict

from src.core.orchestrator import create_orchestrator_graph
from src.core.researcher_config import (
    get_agent_config,
    get_llm_config,
    get_mcp_config,
    get_research_config,
)

logger = logging.getLogger(__name__)


def _autopilot_mode_enabled(context: Dict[str, Any] | None = None) -> bool:
    """Return whether autonomous runs should avoid interactive clarification waits."""
    if context and "autopilot_mode" in context:
        return bool(context["autopilot_mode"])

    explicit = os.getenv("SPARKLEFORGE_AUTOPILOT_MODE")
    if explicit is not None:
        return explicit.lower() not in {"0", "false", "no", "off"}

    # Default to autonomous execution. Interactive clarification must be explicitly enabled
    # by setting SPARKLEFORGE_AUTOPILOT_MODE=false.
    return True


class AutonomousOrchestrator:
    """Modularized LangGraph Orchestrator delegating to specialized nodes."""

    def __init__(self):
        """초기화 및 의존성 주입."""
        self.llm_config = get_llm_config()
        self.agent_config = get_agent_config()
        self.research_config = get_research_config()
        self.mcp_config = get_mcp_config()

        # 스트리밍 매니저
        from src.core.streaming_manager import get_streaming_manager

        self.streaming_manager = get_streaming_manager()

        # 의존 시스템
        from src.agents.creativity_agent import CreativityAgent
        from src.core.adaptive_research_depth import AdaptiveResearchDepth
        from src.core.context_loader import ContextLoader
        from src.core.recursive_context_manager import get_recursive_context_manager
        from src.storage.hybrid_storage import HybridStorage

        self.hybrid_storage = HybridStorage()
        self.creativity_agent = CreativityAgent()
        self.context_loader = ContextLoader()
        self.context_manager = get_recursive_context_manager()

        # Research Depth
        depth_config = (
            self.research_config.research_depth
            if hasattr(self.research_config, "research_depth")
            else {}
        )
        self.research_depth = AdaptiveResearchDepth(depth_config)

        # Graph assembly
        self.graph = create_orchestrator_graph(
            creativity_agent=self.creativity_agent,
            context_manager=self.context_manager,
            streaming_manager=self.streaming_manager,
            hybrid_storage=self.hybrid_storage,
            context_loader=self.context_loader,
            research_depth=self.research_depth,
            llm_config=self.llm_config,
            agent_config=self.agent_config,
        )
        self.graph.recursion_limit = 100

    async def execute(
        self, request: str, context: Dict[str, Any] = None, objective_id: str = None
    ) -> Dict[str, Any]:
        """연구 실행 워크플로우 기동.

        Args:
            objective_id: 기존 실행을 재개하려면 이전에 사용된 objective_id를 전달.
                생략하면 새 objective_id를 생성해 새 실행을 시작.
        """
        resuming = objective_id is not None
        objective_id = objective_id or f"research_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        config = {"configurable": {"thread_id": objective_id}}

        try:
            if resuming:
                checkpoint = await self.graph.aget_state(config)
                if checkpoint and checkpoint.values:
                    logger.info(f"↩️  Resuming orchestrator run '{objective_id}' from checkpoint")
                    final_state = await self.graph.ainvoke(None, config)
                    return final_state
                logger.warning(
                    f"No checkpoint found for objective_id='{objective_id}', starting fresh"
                )

            logger.info(f"🚀 Starting modularized autonomous research: {request[:50]}...")
            initial_state = {
                "user_request": request,
                "context": context or {},
                "autopilot_mode": _autopilot_mode_enabled(context),
                "objective_id": objective_id,
                "iteration": 0,
                "max_iterations": 10,
                "should_continue": True,
                "current_step": "analyze_objectives",
                "innovation_stats": {},
                "messages": [],
            }
            final_state = await self.graph.ainvoke(initial_state, config)
            return final_state
        except Exception as e:
            logger.error(f"❌ Orchestrator execution failed: {e}")
            return {"error": str(e), "success": False}

    async def run_research(
        self,
        user_request: str,
        context: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """Legacy alias for execute()."""
        return await self.execute(user_request, context)

    def ensure_legacy_langgraph_workflow(self) -> None:
        """Backward compatibility helper.

        The current orchestrator builds its graph during initialization, so this
        method intentionally has no side effects.
        """
