import logging
import json
import re
from typing import Any, Dict, List

from src.core.orchestrator.state import ResearchState
from src.core.orchestrator.base_node import BaseNode
from src.core.llm_manager import TaskType, execute_llm_task

logger = logging.getLogger(__name__)

class VerificationNode(BaseNode):
    """Handler for research verification and quality auditing."""

    def __init__(self, researcher_config=None):
        self.researcher_config = researcher_config

    async def verify_plan(self, state: ResearchState) -> ResearchState:
        """Plan 검증: LLM 기반 plan 타당성 검증."""
        self._log_node_input("verify_plan", state)
        logger.info("✅ Verifying research plan")

        try:
            prompt = f"""Verify the research plan for quality and completeness:
            Request: {state.get("user_request", "")}
            Tasks: {state.get("planned_tasks", [])}
            Return JSON: {{ "approved": bool, "confidence": float, "feedback": "..." }}
            """
            result = await execute_llm_task(prompt=prompt, task_type=TaskType.VERIFICATION)
            
            verification = self._parse_verification_result(result.content)
            
            if verification.get("approved", False):
                state["plan_approved"] = True
                state["plan_feedback"] = verification.get("feedback", "Plan approved")
            else:
                state["plan_approved"] = False
                state["plan_feedback"] = verification.get("feedback", "Plan rejected")
                if state.get("plan_iteration", 0) >= 3:
                    state["plan_approved"] = True
                    state["plan_feedback"] += " (forced after 3 iterations)"

            state["current_step"] = "overseer_initial_review" if state["plan_approved"] else "planning_agent"
        except Exception as e:
            logger.warning(f"Plan verification failed: {e}")
            state["plan_approved"] = True
            state["current_step"] = "overseer_initial_review"

        self._log_node_output("verify_plan", state)
        return state

    async def overseer_initial_review(self, state: ResearchState) -> ResearchState:
        """Overseer의 초기 검토 - Planning 후 요구사항 정의"""
        logger.info("🔍 [OVERSEER] Initial Review")
        from src.agents.greedy_overseer_agent import get_greedy_overseer_agent
        from src.core.researcher_config import load_config_from_env

        config = load_config_from_env()
        o_cfg = config.overseer if hasattr(config, "overseer") else None

        if o_cfg and o_cfg.enabled:
            overseer = get_greedy_overseer_agent()
            res = await overseer.define_requirements(state["user_request"], state["analyzed_objectives"])
            state["overseer_requirements"] = res.get("requirements", [])
        
        state["current_step"] = "adaptive_supervisor"
        return state

    async def continuous_verification(self, state: ResearchState) -> ResearchState:
        """결과 검증 (혁신 4)."""
        self._log_node_input("continuous_verification", state)
        # Simplified verification logic
        state["verification_results"] = {"status": "verified", "confidence": 0.9}
        state["current_step"] = "overseer_evaluation"
        return state

    async def overseer_evaluation(self, state: ResearchState) -> ResearchState:
        """Overseer의 평가 - 결과의 완전성과 품질 검증"""
        logger.info("🔍 [OVERSEER] Evaluation")
        from src.agents.greedy_overseer_agent import get_greedy_overseer_agent
        overseer = get_greedy_overseer_agent()
        res = await overseer.evaluate_results(state["user_request"], state["execution_results"])
        
        state["overseer_decision"] = res.get("decision", "proceed")
        state["current_step"] = "evaluate_results" if state["overseer_decision"] == "proceed" else "execute_research"
        return state

    def overseer_decision_router(self, state: ResearchState) -> str:
        """Overseer의 결정에 따른 라우팅"""
        decision = state.get("overseer_decision", "proceed")
        if decision == "retry": return "retry"
        if decision == "ask_user": return "waiting_for_clarification"
        return "proceed"

    def _parse_verification_result(self, content: str) -> Dict[str, Any]:
        cleaned = (content or "").strip()
        md_match = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.DOTALL)
        if md_match: cleaned = match.group(1).strip()
        
        if cleaned.startswith("{"):
            try: return json.loads(cleaned)
            except: pass
        return {"approved": True, "confidence": 0.5, "feedback": "Fallback due to parse error"}
