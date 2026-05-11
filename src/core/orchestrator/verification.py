import asyncio
import json
import logging
import re
from typing import Any, Dict

from src.core.llm_manager import TaskType, execute_llm_task
from src.core.orchestrator.base_node import BaseNode
from src.core.orchestrator.state import ResearchState

logger = logging.getLogger(__name__)

# 검증 재시도 설정
_VERIFY_MAX_RETRIES = 3
_VERIFY_BASE_BACKOFF = 1.0


class VerificationNode(BaseNode):
    """Handler for research verification and quality auditing."""

    def __init__(self, researcher_config=None):
        self.researcher_config = researcher_config

    async def verify_plan(self, state: ResearchState) -> ResearchState:
        """Plan 검증: LLM 기반 plan 타당성 검증.

        강제 approve 없이, LLM이 reject하면 피드백을 포함해 plan을 재생성합니다.
        max_iterations 이내에서 계속 재시도하며, LLM 파싱 실패 시에도 retry합니다.
        """
        self._log_node_input("verify_plan", state)
        logger.info("✅ Verifying research plan")

        plan_iteration = state.get("plan_iteration", 0)
        planned_tasks = state.get("planned_tasks", [])
        user_request = state.get("user_request", "")

        # 이전 피드백이 있으면 포함
        previous_feedback = state.get("plan_feedback", "")
        feedback_context = ""
        if plan_iteration > 0 and previous_feedback:
            feedback_context = f"\n\nPrevious verification feedback (iteration {plan_iteration}):\n{previous_feedback}\n"

        prompt = f"""Verify the following research plan for quality and completeness.

User Request: {user_request}
{feedback_context}
Planned Tasks (count: {len(planned_tasks)}):
{json.dumps(planned_tasks[:10], ensure_ascii=False, indent=2)[:3000]}

Evaluate the plan on:
1. Coverage: Does the plan cover all aspects of the user's request?
2. Feasibility: Are the tasks actionable and well-defined?
3. Completeness: Are there missing critical tasks?
4. Coherence: Do the tasks form a logical research workflow?

Return ONLY a JSON object:
{{ "approved": true/false, "confidence": 0.0-1.0, "feedback": "detailed feedback explaining your decision" }}
"""

        last_error = None
        for attempt in range(_VERIFY_MAX_RETRIES):
            try:
                result = await execute_llm_task(prompt=prompt, task_type=TaskType.VERIFICATION)
                verification = self._parse_verification_result(result.content)

                if verification is not None:
                    if verification.get("approved", False):
                        state["plan_approved"] = True
                        state["plan_feedback"] = verification.get("feedback", "Plan approved")
                    else:
                        state["plan_approved"] = False
                        state["plan_feedback"] = verification.get("feedback", "Plan needs revision")

                    state["current_step"] = (
                        "overseer_initial_review" if state["plan_approved"] else "planning_agent"
                    )
                    self._log_node_output(
                        "verify_plan",
                        state,
                        {
                            "approved": state["plan_approved"],
                            "iteration": plan_iteration,
                        },
                    )
                    return state

                last_error = "LLM returned unparseable response"
                logger.warning(
                    f"verify_plan: attempt {attempt + 1}/{_VERIFY_MAX_RETRIES} — unparseable response"
                )

            except Exception as e:
                last_error = str(e)
                logger.warning(
                    f"verify_plan: attempt {attempt + 1}/{_VERIFY_MAX_RETRIES} failed — {e}"
                )

            if attempt < _VERIFY_MAX_RETRIES - 1:
                backoff = _VERIFY_BASE_BACKOFF * (2**attempt)
                await asyncio.sleep(backoff)

        # 모든 retry 실패 시 에러 전파
        state["plan_approved"] = False
        state["plan_feedback"] = (
            f"Verification failed after {_VERIFY_MAX_RETRIES} attempts: {last_error}"
        )
        state["current_step"] = "planning_agent"
        state["error_message"] = state["plan_feedback"]
        logger.error(f"verify_plan: all {_VERIFY_MAX_RETRIES} attempts failed — {last_error}")

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
            res = await overseer.define_requirements(
                state["user_request"], state["analyzed_objectives"]
            )
            state["overseer_requirements"] = res.get("requirements", [])

        state["current_step"] = "adaptive_supervisor"
        return state

    async def continuous_verification(self, state: ResearchState) -> ResearchState:
        """결과 검증 (혁신 4) — LLM 기반 결과 품질 검증."""
        self._log_node_input("continuous_verification", state)

        execution_results = state.get("execution_results", [])
        user_request = state.get("user_request", "")

        prompt = f"""Verify the quality and relevance of the following research results.

User Request: {user_request}

Execution Results (count: {len(execution_results)}):
{json.dumps(execution_results[:5], ensure_ascii=False, indent=2)[:3000]}

Evaluate:
1. Relevance: Do the results address the user's request?
2. Quality: Is the data accurate and well-sourced?
3. Completeness: Are there significant gaps?

Return ONLY a JSON object:
{{ "status": "verified" or "needs_improvement", "confidence": 0.0-1.0, "issues": ["issue1", ...] }}
"""

        try:
            result = await execute_llm_task(prompt=prompt, task_type=TaskType.VERIFICATION)
            verification = self._parse_verification_result(result.content)

            if verification is not None:
                state["verification_results"] = {
                    "status": verification.get("status", "verified"),
                    "confidence": verification.get("confidence", 0.5),
                    "issues": verification.get("issues", []),
                }
            else:
                raise RuntimeError("Verification result unparseable")

        except Exception as e:
            logger.error(f"Continuous verification failed: {e}")
            raise

        state["current_step"] = "overseer_evaluation"
        return state

    async def overseer_evaluation(self, state: ResearchState) -> ResearchState:
        """Overseer의 평가 - 결과의 완전성과 품질 검증"""
        logger.info("🔍 [OVERSEER] Evaluation")
        from src.agents.greedy_overseer_agent import get_greedy_overseer_agent

        overseer = get_greedy_overseer_agent()
        res = await overseer.evaluate_results(state["user_request"], state["execution_results"])

        state["overseer_decision"] = res.get("decision", "proceed")
        state["current_step"] = (
            "evaluate_results" if state["overseer_decision"] == "proceed" else "execute_research"
        )
        return state

    def overseer_decision_router(self, state: ResearchState) -> str:
        """Overseer의 결정에 따른 라우팅"""
        decision = state.get("overseer_decision", "proceed")
        if decision == "retry":
            return "retry"
        if decision == "ask_user":
            return "waiting_for_clarification"
        return "proceed"

    def _parse_verification_result(self, content: str) -> Dict[str, Any] | None:
        """LLM 응답에서 JSON을 파싱합니다. 실패 시 None을 반환합니다."""
        cleaned = (content or "").strip()

        # Markdown 코드 블록 제거
        md_match = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.DOTALL)
        if md_match:
            cleaned = md_match.group(1).strip()

        if cleaned.startswith("{"):
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                logger.warning("JSON decode failed in verification result")
                return None

        logger.warning("Verification result is not JSON")
        return None
