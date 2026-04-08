import logging
import json
from typing import Any, Dict
from src.core.orchestrator.state import ResearchState

logger = logging.getLogger(__name__)

class BaseNode:
    """Base handler for Orchestrator nodes."""

    def _log_node_input(self, node_name: str, state: ResearchState):
        """노드 입력 로깅."""
        logger.info(f"\\n{'=' * 80}\\n🔵 NODE INPUT: {node_name}\\n{'=' * 80}")
        logger.info(f"User Request: {state.get('user_request', 'N/A')}")
        logger.info(f"Current Step: {state.get('current_step', 'N/A')}")
        logger.info(f"Iteration: {state.get('iteration', 0)}")
        logger.info(f"Complexity Score: {state.get('complexity_score', 'N/A')}")

        # 주요 필드 선택적 로깅
        if "analyzed_objectives" in state:
            logger.info(
                f"Objectives Count: {len(state.get('analyzed_objectives', []))}"
            )
        if "planned_tasks" in state:
            logger.info(f"Planned Tasks Count: {len(state.get('planned_tasks', []))}")
        if "agent_assignments" in state:
            logger.info(
                f"Agent Assignments Count: {len(state.get('agent_assignments', {}))}"
            )
        logger.info("=" * 80)

    def _log_node_output(
        self, node_name: str, state: ResearchState, key_changes: Dict[str, Any] = None
    ):
        """노드 출력 로깅."""
        logger.info(f"\\n{'=' * 80}\\n🟢 NODE OUTPUT: {node_name}\\n{'=' * 80}")
        logger.info(f"Next Step: {state.get('current_step', 'N/A')}")
        logger.info(f"Should Continue: {state.get('should_continue', 'N/A')}")
        logger.info(f"Error Message: {state.get('error_message', 'None')}")

        # 주요 변경사항 로깅
        if key_changes:
            logger.info(
                f"Key Changes:\\n{json.dumps(key_changes, indent=2, ensure_ascii=False)}"
            )

        # State 업데이트 요약
        logger.info(f"Complexity Score: {state.get('complexity_score', 'N/A')}")
        logger.info(
            f"Allocated Researchers: {state.get('allocated_researchers', 'N/A')}"
        )
        logger.info(f"Iteration: {state.get('iteration', 0)}")
        logger.info("=" * 80)
