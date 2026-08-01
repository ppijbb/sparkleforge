"""Forge Master Controller - Central Master Controller for External AI Agents

SparkleForge 중앙 에이전트 OS의 주 제어 장치.
외부 AI CLI 도구(Claude Code, Codex, Gemini CLI, Hermes 등)를 상시(24/7) 관제하고,
토큰 최소화, 세션 및 문맥 유지, 적대적 검증 게이트를 통합 총괄함.

이 컨트롤러는 선택된 에이전트가 실패했다고 해서 코드가 알아서 다른 CLI로
갈아타지 않는다 (ESCALATE_TO_FALLBACK이어도 자동 전환 없음). 어떤 에이전트를
쓸지, 실패 후 다른 에이전트로 다시 시도할지는 실제 판단 주체
(agent_loop의 tool-call 턴 - src/core/forge_master/tools.py의
`dispatch_batch_to_forge_master` 참고, 또는 명시적 preferred_agent를 넘기는 호출자)의
몫이며, 여기서는 그 판단 결과(agent_name)를 실행하고 실패 시 후보 목록만
정보로 돌려줄 뿐이다.
"""

import logging
from typing import Any, Dict, List, Optional

from .adversarial_evaluator import AdversarialEvaluator
from .personas import apply_persona
from .router import ForgeMasterRouter
from .session_manager import ForgeMasterSessionManager
from .token_minimizer import TokenMinimizer

logger = logging.getLogger(__name__)


class ForgeMasterController:
    """Forge Master 중앙 컨트롤러"""

    def __init__(
        self,
        router: Optional[ForgeMasterRouter] = None,
        token_minimizer: Optional[TokenMinimizer] = None,
        session_manager: Optional[ForgeMasterSessionManager] = None,
        adversarial_evaluator: Optional[AdversarialEvaluator] = None,
    ):
        self.router = router or ForgeMasterRouter()
        self.token_minimizer = token_minimizer or TokenMinimizer()
        self.session_manager = session_manager or ForgeMasterSessionManager()
        self.adversarial_evaluator = adversarial_evaluator or AdversarialEvaluator()

    async def execute_task_with_master_control(
        self,
        task_query: str,
        context: Optional[str] = None,
        required_capabilities: Optional[List[str]] = None,
        preferred_agent: Optional[str] = None,
        is_persistent_session: bool = False,
        session_id: Optional[str] = None,
        max_retries: int = 2,
        persona: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Forge Master 실행 파이프라인

        1. Goal 부여 (agent_name이 없으면 비-에이전트 기본값으로 휴리스틱 라우팅)
        2. 프롬프트 및 문맥 압축 (Token Minimization)
        3. 세션 환경 하 CLI 에이전트 구동 (Sub-Agent or Multi-Session)
        4. 결과물에 대한 극단적 적대적 평가 및 검증 (Adversarial Audit)
        5. 같은 에이전트 재시도(피드백 반영)까지만 자동 수행 - 다른 CLI로의
           전환은 하지 않으며, 실패 시 폴백 후보만 정보로 반환한다

        Args:
            task_query: 실행할 요청 쿼리
            context: 추가 문맥
            required_capabilities: 요구되는 에이전트 역량
            preferred_agent: 사용할 에이전트 이름. 실제 판단 주체가 명시적으로
                고른 값이어야 하며, 생략 시에만 route_task 휴리스틱 기본값을 쓴다.
            is_persistent_session: 24/7 멀티 세션 지원 여부
            session_id: 기존 세션 ID (있을 경우 재사용)
            max_retries: 같은 에이전트에 대한 최대 재시도 횟수
            **kwargs: 추가 전달 옵션

        Returns:
            총괄 수집 및 검증된 결과 딕셔너리. 실패 시 `fallback_candidates`에
            (자동 실행되지 않은) 대안 에이전트 후보를 담아 돌려준다.
        """
        logger.info(f"ForgeMaster Controller starting task: '{task_query[:60]}...'")

        # 1. Goal 부여. agent_name을 명시하지 않은 호출자에게만 결정론적 기본값을 골라준다.
        assignment = self.router.route_task(
            task_description=task_query,
            required_capabilities=required_capabilities,
            preferred_agent=preferred_agent,
        )
        current_agent = assignment.agent_name
        assigned_goal = assignment.assigned_goal
        fallback_candidates = assignment.fallback_agents.copy()

        # 2. 토큰 최소화 및 문맥 압축
        compact_prompt = self.token_minimizer.compact_prompt(assigned_goal)
        compact_ctx = self.token_minimizer.compact_prompt(context or "")

        # 3. 세션 확보
        if not session_id:
            session = self.session_manager.create_session(
                agent_name=current_agent, is_persistent=is_persistent_session
            )
            session_id = session.session_id

        attempt = 0
        last_error = ""

        while attempt <= max_retries:
            attempt += 1
            logger.info(
                f"ForgeMaster executing with agent '{current_agent}' (attempt {attempt}/{max_retries + 1})"
            )

            # 4. CLI 에이전트 구동
            exec_result = await self.session_manager.execute_in_session(
                session_id=session_id,
                query=compact_prompt,
                compact_context=compact_ctx,
                **kwargs,
            )

            # 5. 적대적 평가 수행 (Zero-Trust Adversarial Audit)
            adv_audit = await self.adversarial_evaluator.evaluate_output(
                task_query=task_query,
                agent_name=current_agent,
                execution_result=exec_result,
            )

            if adv_audit.passed:
                # 적대적 검증 통과 -> 결과 응축 후 반환
                distilled = self.token_minimizer.distill_response(
                    exec_result.get("response", "")
                )
                reduction_metrics = self.token_minimizer.estimate_token_reduction(
                    context or "", compact_ctx
                )

                return {
                    "success": True,
                    "master_verdict": "PASSED",
                    "agent_used": current_agent,
                    "session_id": session_id,
                    "response": distilled,
                    "raw_response": exec_result.get("response", ""),
                    "adversarial_audit": {
                        "passed": True,
                        "skepticism_score": adv_audit.skepticism_score,
                        "feedback": adv_audit.adversarial_feedback,
                    },
                    "token_metrics": reduction_metrics,
                    "attempts": attempt,
                }

            # 검증 실패 처리. 다른 CLI로 코드가 알아서 전환하지 않는다:
            # ESCALATE_TO_FALLBACK(치명적 실패)이면 즉시 중단하고 반환하며,
            # 그 외엔 같은 에이전트에 피드백을 실어 재시도만 한다. 정말 다른
            # 에이전트가 필요한 판단은 호출자(에이전트) 몫이므로 fallback_candidates에
            # 후보만 정보로 실어 반환한다.
            logger.warning(
                f"Adversarial Audit failed for agent '{current_agent}' (verdict={adv_audit.verdict}): {adv_audit.adversarial_feedback}"
            )
            last_error = adv_audit.adversarial_feedback

            if adv_audit.verdict == "ESCALATE_TO_FALLBACK":
                break

            compact_prompt = (
                f"{compact_prompt}\n\n[REJECTION FEEDBACK]: {adv_audit.adversarial_feedback}. Please fix immediately."
            )

        # 재시도 소진 또는 ESCALATE_TO_FALLBACK으로 중단된 경우.
        # 다른 CLI로의 전환은 여기서 하지 않고, 후보만 알려준다.
        return {
            "success": False,
            "master_verdict": "REJECTED",
            "last_agent_used": current_agent,
            "session_id": session_id,
            "error": f"Adversarial Evaluation failed after {attempt} attempt(s) with '{current_agent}'. Last error: {last_error}",
            "response": "",
            "attempts": attempt,
            "fallback_candidates": fallback_candidates,
        }
