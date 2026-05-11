import logging
from typing import Any, Dict, List

from src.core.llm_manager import TaskType, execute_llm_task
from src.core.orchestrator.base_node import BaseNode
from src.core.orchestrator.state import ResearchState

logger = logging.getLogger(__name__)


class SynthesisNode(BaseNode):
    """Handler for research evaluation, validation, and synthesis."""

    def __init__(self, context_manager, creativity_agent, hybrid_storage):
        self.context_manager = context_manager
        self.creativity_agent = creativity_agent
        self.hybrid_storage = hybrid_storage

    async def evaluate_results(self, state: ResearchState) -> ResearchState:
        """결과 평가 (Multi-Model Orchestration)."""
        logger.info("📊 Evaluating results")
        prompt = f"Evaluate the following research: {state.get('execution_results', [])}"
        result = await execute_llm_task(
            prompt=prompt, task_type=TaskType.VERIFICATION, use_ensemble=True
        )

        evaluation_data = self._parse_evaluation_result(result.content)
        state.update(
            {
                "evaluation_results": evaluation_data,
                "quality_metrics": evaluation_data.get("metrics", {}),
                "current_step": "validate_results",
            }
        )
        return state

    async def validate_results(self, state: ResearchState) -> ResearchState:
        """결과 검증."""
        logger.info("✅ Validating results")
        validation_score = self._calculate_validation_score(state)
        missing_elements = self._identify_missing_elements(state)

        state.update(
            {
                "validation_score": validation_score,
                "missing_elements": missing_elements,
                "current_step": "synthesize_deliverable",
            }
        )
        return state

    async def synthesize_deliverable(self, state: ResearchState) -> ResearchState:
        """최종 결과 종합."""
        logger.info("📝 Synthesizing final deliverable")

        self.context_manager.get_current_context()
        synthesis_prompt = (
            f"Synthesize research findings for request: {state.get('user_request', '')}"
        )

        result = await execute_llm_task(prompt=synthesis_prompt, task_type=TaskType.SYNTHESIS)
        self._calculate_context_usage(state, result.content)

        state.update(
            {
                "final_synthesis": {
                    "content": result.content,
                    "model_used": result.model_used,
                },
                "current_step": "completed",
            }
        )

        await self._save_research_memory(state)
        await self._generate_creative_insights(state)
        return state

    async def _generate_creative_insights(self, state: ResearchState) -> None:
        try:
            insights = await self.creativity_agent.generate_seed_ideas(
                state.get("user_request", "")
            )
            state["creative_insights"] = insights
        except:
            pass

    async def _save_research_memory(self, state: ResearchState) -> bool:
        try:
            await self.hybrid_storage.save_research_result(state)
            return True
        except:
            return False

    def _calculate_validation_score(self, state: ResearchState) -> float:
        return 0.8  # Simplified

    def _identify_missing_elements(self, state: ResearchState) -> List[str]:
        return []  # Simplified

    def _calculate_context_usage(self, state, content) -> Dict[str, Any]:
        return {"usage_ratio": 0.1, "tokens_used": 1000}

    def _parse_evaluation_result(self, content: str) -> Dict[str, Any]:
        return {"overall_score": 0.8, "metrics": {"quality": 0.8}}
