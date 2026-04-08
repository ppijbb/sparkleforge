"""LangGraph Orchestrator (v3.0 - Modularized Architecture)

Modular architecture refactored from the monolithic 165KB orchestrator.
Delegates core logic to src.core.orchestrator packages.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List

from src.core.orchestrator import ResearchState, create_orchestrator_graph
from src.core.researcher_config import (
    get_agent_config,
    get_llm_config,
    get_mcp_config,
    get_research_config,
)

logger = logging.getLogger(__name__)

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
        from src.storage.hybrid_storage import HybridStorage
        from src.core.context_loader import ContextLoader
        from src.core.recursive_context_manager import get_recursive_context_manager
        from src.core.adaptive_research_depth import AdaptiveResearchDepth

        self.hybrid_storage = HybridStorage()
        self.creativity_agent = CreativityAgent()
        self.context_loader = ContextLoader()
        self.context_manager = get_recursive_context_manager()
        
        # Research Depth
        depth_config = self.research_config.research_depth if hasattr(self.research_config, "research_depth") else {}
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
            agent_config=self.agent_config
        )

    async def execute(self, request: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """연구 실행 워크플로우 기동."""
        logger.info(f"🚀 Starting modularized autonomous research: {request[:50]}...")
        
        initial_state = {
            "user_request": request,
            "context": context or {},
            "objective_id": f"research_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "iteration": 0,
            "max_iterations": 10,
            "should_continue": True,
            "current_step": "analyze_objectives",
            "innovation_stats": {},
            "messages": []
        }

        try:
            final_state = await self.graph.ainvoke(initial_state)
            return final_state
        except Exception as e:
            logger.error(f"❌ Orchestrator execution failed: {e}")
            return {"error": str(e), "success": False}

    def ensure_legacy_langgraph_workflow(self) -> None:
        """Backward compatibility helper."""
        pass
