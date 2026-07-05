"""진행률 정량 추적기 - 체크리스트 상태를 가중치 기반으로 추적 (M3)."""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List

from src.core.anvil.dynamic_checklist_generator import Checklist, ChecklistItem, ItemStatus

logger = logging.getLogger(__name__)


@dataclass
class ProgressSnapshot:
    """특정 시점의 진행률."""

    completion_ratio: float
    completed: int
    failed: int
    pending: int
    timestamp: float = field(default_factory=time.time)


class ProgressTracker:
    """체크리스트 항목 상태 전이를 기록하고 가중 완료율을 계산한다."""

    def __init__(self, checklist: Checklist):
        self.checklist = checklist
        self.history: List[ProgressSnapshot] = []

    def mark(self, item_id: str, status: ItemStatus) -> ChecklistItem:
        """항목 상태 전이 + 스냅샷 기록."""
        item = self._find(item_id)
        if status == ItemStatus.FAILED:
            item.failure_count += 1
        item.status = status
        snapshot = self.snapshot()
        logger.info(
            "Checklist %s: item %s -> %s (completion %.0f%%)",
            self.checklist.checklist_id,
            item_id,
            status.value,
            snapshot.completion_ratio * 100,
        )
        return item

    def snapshot(self) -> ProgressSnapshot:
        """가중치 기반 완료율 스냅샷."""
        items = self.checklist.items
        total_weight = sum(i.weight for i in items) or 1.0
        done_weight = sum(
            i.weight for i in items if i.status in (ItemStatus.COMPLETED, ItemStatus.SKIPPED)
        )
        snap = ProgressSnapshot(
            completion_ratio=round(done_weight / total_weight, 4),
            completed=sum(1 for i in items if i.status == ItemStatus.COMPLETED),
            failed=sum(1 for i in items if i.status == ItemStatus.FAILED),
            pending=sum(1 for i in items if i.status == ItemStatus.PENDING),
        )
        self.history.append(snap)
        return snap

    def is_complete(self) -> bool:
        return all(
            i.status in (ItemStatus.COMPLETED, ItemStatus.SKIPPED) for i in self.checklist.items
        )

    def failed_items(self) -> List[ChecklistItem]:
        return [i for i in self.checklist.items if i.status == ItemStatus.FAILED]

    def summary(self) -> Dict[str, object]:
        snap = self.snapshot()
        return {
            "checklist_id": self.checklist.checklist_id,
            "domain": self.checklist.domain,
            "completion_ratio": snap.completion_ratio,
            "completed": snap.completed,
            "failed": snap.failed,
            "pending": snap.pending,
            "items": [
                {
                    "id": i.item_id,
                    "description": i.description,
                    "status": i.status.value,
                    "failures": i.failure_count,
                }
                for i in self.checklist.items
            ],
        }

    def _find(self, item_id: str) -> ChecklistItem:
        for item in self.checklist.items:
            if item.item_id == item_id:
                return item
        raise KeyError(f"Checklist item not found: {item_id}")
