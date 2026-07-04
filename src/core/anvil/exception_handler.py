"""체크리스트 예외 대응기 - 실패 항목에 대한 동적 대응 결정 (M3)."""

import logging
from enum import Enum

from src.core.anvil.dynamic_checklist_generator import ChecklistItem

logger = logging.getLogger(__name__)


class RecoveryAction(Enum):
    RETRY = "retry"  # 같은 방식으로 재시도
    REPLAN = "replan"  # 접근 방식을 바꿔 재계획
    SKIP = "skip"  # 비필수 항목 건너뛰기
    ABORT = "abort"  # 진행 불가


class ChecklistExceptionHandler:
    """항목 실패 횟수와 가중치를 바탕으로 다음 대응을 결정한다."""

    def __init__(self, max_retries: int = 2, max_replans: int = 1):
        self.max_retries = max_retries
        self.max_replans = max_replans

    def decide(self, item: ChecklistItem) -> RecoveryAction:
        failures = item.failure_count
        if failures <= self.max_retries:
            action = RecoveryAction.RETRY
        elif failures <= self.max_retries + self.max_replans:
            action = RecoveryAction.REPLAN
        elif item.weight < 1.0:
            action = RecoveryAction.SKIP
        else:
            action = RecoveryAction.ABORT

        logger.info(
            "Recovery decision for item %s (failures=%d, weight=%.1f): %s",
            item.item_id,
            failures,
            item.weight,
            action.value,
        )
        return action
