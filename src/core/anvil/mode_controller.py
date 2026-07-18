"""실행 모드 컨트롤러 - 자율 모드와 HITL 협업 모드 간 동적 전환 (M5)."""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List

logger = logging.getLogger(__name__)


class ExecutionMode(Enum):
    AUTONOMOUS = "autonomous"  # 사람 개입 없이 자율 실행
    HITL_COLLABORATIVE = "hitl_collaborative"  # 단계별 사용자 협업 실행
    PLAN_FIRST = "plan_first"  # 계획 승인 전까지 쓰기 액션 차단


@dataclass
class ModeTransition:
    """모드 전환 기록."""

    from_mode: ExecutionMode
    to_mode: ExecutionMode
    reason: str
    timestamp: float = field(default_factory=time.time)


class ModeController:
    """실행 중 수집되는 신호를 바탕으로 자율/협업 모드를 동적으로 전환한다.

    협업 모드로 전환하는 신호:
        - 연속 실패가 임계치를 넘음 (자율 복구 한계)
        - 의도 가드레일이 사람 확인 필요를 보고함
        - 체크포인트에서 사용자가 경로 수정(revise)이나 중단(abort)을 결정함
        - 필요한 방법을 끝내 확보하지 못함 (unresolved capability)

    자율 모드로 복귀하는 신호:
        - 연속 성공이 임계치에 도달해 안정성이 회복됨

    M4/M5 모듈과 타입 결합 없이 문자열·불리언 신호만 받으므로
    어떤 워크플로우 계층에서도 독립적으로 사용할 수 있다.
    """

    def __init__(
        self,
        initial_mode: ExecutionMode = ExecutionMode.AUTONOMOUS,
        failure_threshold: int = 3,
        recovery_threshold: int = 3,
        plan_first: bool = False,
    ):
        self.mode = initial_mode
        self.failure_threshold = max(1, failure_threshold)
        self.recovery_threshold = max(1, recovery_threshold)
        self.transitions: List[ModeTransition] = []
        self._consecutive_failures = 0
        self._consecutive_successes = 0
        if plan_first and initial_mode != ExecutionMode.PLAN_FIRST:
            self._switch(
                ExecutionMode.PLAN_FIRST,
                "세션 시작 시 --plan 선언으로 계획 우선 모드 진입",
            )
        self.plan_approved = False
        self.plan_revisions = 0

    # --- 신호 수신 ---

    def record_success(self) -> ExecutionMode:
        """태스크/항목 성공 신호. 협업 모드에서 안정이 회복되면 자율 복귀."""
        self._consecutive_failures = 0
        self._consecutive_successes += 1
        if (
            self.mode == ExecutionMode.HITL_COLLABORATIVE
            and self._consecutive_successes >= self.recovery_threshold
        ):
            self._switch(
                ExecutionMode.AUTONOMOUS,
                f"연속 성공 {self._consecutive_successes}회로 안정성 회복",
            )
        return self.mode

    def record_failure(self) -> ExecutionMode:
        """태스크/항목 실패 신호. 연속 실패가 임계치를 넘으면 협업 전환."""
        self._consecutive_successes = 0
        self._consecutive_failures += 1
        if (
            self.mode == ExecutionMode.AUTONOMOUS
            and self._consecutive_failures >= self.failure_threshold
        ):
            self._switch(
                ExecutionMode.HITL_COLLABORATIVE,
                f"연속 실패 {self._consecutive_failures}회로 자율 복구 한계 도달",
            )
        return self.mode

    def on_intent_review_needed(self) -> ExecutionMode:
        """의도 가드레일의 사람 확인 필요 신호 (IntentGuardrail.needs_human_review)."""
        if self.mode == ExecutionMode.AUTONOMOUS:
            self._switch(
                ExecutionMode.HITL_COLLABORATIVE, "의도 이탈 감지로 사람 확인 필요"
            )
        return self.mode

    def on_checkpoint_decision(self, decision: str) -> ExecutionMode:
        """체크포인트 결정 신호 ('approve' | 'revise' | 'abort')."""
        if decision in ("revise", "abort") and self.mode == ExecutionMode.AUTONOMOUS:
            self._switch(
                ExecutionMode.HITL_COLLABORATIVE,
                f"체크포인트에서 사용자 개입 결정: {decision}",
            )
        return self.mode

    # --- 계획 우선 모드 (Plan Mode) ---

    def is_write_blocked(self) -> bool:
        """PLAN_FIRST 모드에서 계획이 승인되기 전까지 쓰기 액션 차단 여부."""
        return self.is_plan_first() and not self.plan_approved

    def submit_plan(self, approved: bool, feedback: str = "") -> ExecutionMode:
        """계획 초안에 대한 사람 승인/수정 요청 결과 반영.

        approved=True 이면 AUTONOMOUS 또는 HITL_COLLABORATIVE 로 전환해 실행 시작.
        approved=False 이면 PLAN_FIRST 를 유지하며 계획을 다시 만들어 재승인받는다.
        """
        if self.mode != ExecutionMode.PLAN_FIRST:
            return self.mode

        if approved:
            self.plan_approved = True
            target = (
                ExecutionMode.HITL_COLLABORATIVE
                if feedback.strip()
                else ExecutionMode.AUTONOMOUS
            )
            self._switch(
                target,
                "계획 승인으로 실행 모드 전환" + (f": {feedback}" if feedback.strip() else ""),
            )
        else:
            self.plan_revisions += 1
            self._switch(
                ExecutionMode.PLAN_FIRST,
                f"계획 반려/수정 요청 ({self.plan_revisions}회) - 계획 재작성 후 재승인",
            )
        return self.mode

    def on_unresolved_capability(self, capability: str) -> ExecutionMode:
        """방법 탐색 체인이 끝내 실패한 신호 (MethodResolver UNRESOLVED)."""
        if self.mode == ExecutionMode.AUTONOMOUS:
            self._switch(
                ExecutionMode.HITL_COLLABORATIVE,
                f"'{capability}' 수행 방법을 확보하지 못해 사용자 협의 필요",
            )
        return self.mode

    # --- 상태 조회 ---

    def is_autonomous(self) -> bool:
        return self.mode == ExecutionMode.AUTONOMOUS

    def is_plan_first(self) -> bool:
        return self.mode == ExecutionMode.PLAN_FIRST

    def summary(self) -> Dict[str, object]:
        return {
            "mode": self.mode.value,
            "consecutive_failures": self._consecutive_failures,
            "consecutive_successes": self._consecutive_successes,
            "transitions": [
                {
                    "from": t.from_mode.value,
                    "to": t.to_mode.value,
                    "reason": t.reason,
                }
                for t in self.transitions
            ],
        }

    def _switch(self, to_mode: ExecutionMode, reason: str) -> None:
        transition = ModeTransition(from_mode=self.mode, to_mode=to_mode, reason=reason)
        self.transitions.append(transition)
        logger.info(
            "Execution mode switched: %s -> %s (%s)",
            self.mode.value,
            to_mode.value,
            reason,
        )
        self.mode = to_mode
        self._consecutive_failures = 0
        self._consecutive_successes = 0
