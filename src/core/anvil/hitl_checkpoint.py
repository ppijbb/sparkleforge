"""HITL 체크포인트 - 주요 단계별 사용자 피드백 수집 및 체크리스트 반영 (M4)."""

import asyncio
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from src.core.anvil.dynamic_checklist_generator import Checklist, ChecklistItem
from src.core.anvil.request_analyzer import RequestAnalyzer

logger = logging.getLogger(__name__)


class CheckpointStage(Enum):
    """인터랙티브 피드백을 받는 주요 워크플로우 단계."""

    AFTER_PLANNING = "after_planning"  # 태스크 계획 수립 직후
    AFTER_DATA_COLLECTION = "after_data_collection"  # 주요 데이터 수집 직후
    BEFORE_FINAL_REPORT = "before_final_report"  # 최종 보고서 작성 직전
    INTENT_DRIFT = "intent_drift"  # 의도 가드레일이 이탈을 감지했을 때


class CheckpointDecision(Enum):
    APPROVE = "approve"  # 현재 방향 유지
    REVISE = "revise"  # 피드백을 반영해 경로 수정
    ABORT = "abort"  # 작업 중단


@dataclass
class CheckpointResult:
    """체크포인트 한 번의 처리 결과."""

    stage: CheckpointStage
    decision: CheckpointDecision
    feedback: str = ""
    auto_resolved: bool = False
    checkpoint_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    timestamp: float = field(default_factory=time.time)


# 피드백 제공자: (stage, context) -> (decision, feedback_text)
FeedbackProvider = Callable[
    [CheckpointStage, Dict[str, Any]], Tuple[CheckpointDecision, str]
]


class HITLCheckpointManager:
    """HITL 체크포인트 관리자."""

    def __init__(
        self,
        feedback_provider: FeedbackProvider | None = None,
        checkpoint_dir: str = ".checkpoints",
        timeout_seconds: int = 1800,
    ):
        self.feedback_provider = feedback_provider
        self.history: List[CheckpointResult] = []
        self._analyzer = RequestAnalyzer()
        self.checkpoint_dir = checkpoint_dir
        self.timeout_seconds = timeout_seconds
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    async def checkpoint(
        self, stage: CheckpointStage, context: Dict[str, Any] | None = None
    ) -> CheckpointResult:
        """단계 체크포인트 실행. 제공자가 없으면 자동 승인."""
        context = context or {}

        if self.feedback_provider is None:
            result = CheckpointResult(
                stage=stage, decision=CheckpointDecision.APPROVE, auto_resolved=True
            )
            logger.info("Checkpoint %s auto-approved (headless mode)", stage.value)
            self.history.append(result)
            return result

        try:
            outcome = self.feedback_provider(stage, context)
            if asyncio.iscoroutine(outcome):
                outcome = await outcome
            if outcome is None:
                raise BlockingIOError("Suspension required")
            decision, feedback = outcome
        except Exception as e:
            if isinstance(e, BlockingIOError) or "suspend" in str(e).lower():
                # Save state and exit if provider requires suspension
                checkpoint_id = uuid.uuid4().hex[:8]
                state_file = os.path.join(self.checkpoint_dir, f"{checkpoint_id}.json")
                with open(state_file, "w") as f:
                    json.dump({"stage": stage.value, "context": context, "created_at": time.time()}, f)
                
                logger.info("Checkpoint %s suspended at %s", stage.value, state_file)
                return CheckpointResult(stage=stage, decision=CheckpointDecision.ABORT, checkpoint_id=checkpoint_id)

            logger.warning(
                "Feedback provider failed at %s, defaulting to approve: %s",
                stage.value,
                e,
            )
            decision, feedback = CheckpointDecision.APPROVE, ""

        result = CheckpointResult(stage=stage, decision=decision, feedback=feedback)
        self.history.append(result)
        logger.info(
            "Checkpoint %s resolved: decision=%s feedback_len=%d",
            stage.value,
            decision.value,
            len(feedback),
        )
        return result

    def apply_feedback(
        self, checklist: Checklist, result: CheckpointResult
    ) -> List[ChecklistItem]:
        """REVISE 피드백을 재분석해 체크리스트에 새 항목으로 반영.

        Returns:
            새로 추가된 체크리스트 항목 목록.
        """
        if result.decision != CheckpointDecision.REVISE or not result.feedback.strip():
            return []

        analysis = self._analyzer.analyze(result.feedback)
        added: List[ChecklistItem] = []

        for requirement in analysis.requirements:
            item = ChecklistItem(
                description=f"[피드백] {requirement}",
                success_criteria=f"'{requirement[:60]}' 피드백 요구가 산출물로 확인된다",
            )
            checklist.items.append(item)
            added.append(item)

        for constraint in analysis.constraints:
            item = ChecklistItem(
                description=f"[피드백 제약] {constraint}",
                success_criteria="이후 작업에서 해당 제약을 위반하지 않는다",
                weight=0.5,
            )
            checklist.items.append(item)
            added.append(item)

        logger.info(
            "Feedback applied to checklist %s: %d items added",
            checklist.checklist_id,
            len(added),
        )
        return added

    def aborted(self) -> bool:
        """어느 체크포인트에서든 중단 결정이 있었는지."""
        return any(r.decision == CheckpointDecision.ABORT for r in self.history)

    def summary(self) -> List[Dict[str, Any]]:
        """체크포인트 이력 요약."""
        return [
            {
                "id": r.checkpoint_id,
                "stage": r.stage.value,
                "decision": r.decision.value,
                "auto": r.auto_resolved,
                "feedback": r.feedback,
            }
            for r in self.history
        ]

    def resolve_timeout(self, checkpoint_id: str) -> CheckpointDecision:
        """타임아웃 발생 시 기본 결정 적용."""
        state_file = os.path.join(self.checkpoint_dir, f"{checkpoint_id}.json")
        if os.path.exists(state_file):
            with open(state_file) as f:
                data = json.load(f)
                if time.time() - data.get("created_at", 0) > self.timeout_seconds:
                    os.remove(state_file)
                    logger.info("Checkpoint %s timed out, auto-aborting", checkpoint_id)
                    return CheckpointDecision.ABORT
        return CheckpointDecision.APPROVE
