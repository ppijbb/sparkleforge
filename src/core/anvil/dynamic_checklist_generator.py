"""동적 체크리스트 생성기 - 분석된 요청을 실행 가능한 체크리스트로 변환 (M3)."""

import logging
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List

from src.core.anvil.request_analyzer import RequestAnalysis

logger = logging.getLogger(__name__)


class ItemStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class ChecklistItem:
    """체크리스트 항목: 성공 기준과 가중치를 가진 실행 단위."""

    description: str
    success_criteria: str
    weight: float = 1.0
    status: ItemStatus = ItemStatus.PENDING
    item_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    failure_count: int = 0


@dataclass
class Checklist:
    """요청 하나에 대한 동적 체크리스트."""

    domain: str
    items: List[ChecklistItem] = field(default_factory=list)
    checklist_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])


# 도메인별 기본 마무리 단계
_DOMAIN_FINALIZERS: Dict[str, ChecklistItem] = {
    "code": ChecklistItem(
        description="구현 결과 검증 (테스트/실행 확인)",
        success_criteria="구현 코드가 오류 없이 실행되거나 관련 테스트가 통과한다",
    ),
    "research": ChecklistItem(
        description="수집 정보 종합 및 결과 보고",
        success_criteria="출처가 포함된 종합 결과물이 생성된다",
    ),
    "file": ChecklistItem(
        description="파일 작업 결과 확인",
        success_criteria="대상 파일이 의도한 상태로 존재한다",
    ),
}


class DynamicChecklistGenerator:
    """RequestAnalysis를 정량 추적 가능한 체크리스트로 변환한다."""

    def generate(self, analysis: RequestAnalysis) -> Checklist:
        checklist = Checklist(domain=analysis.domain)

        for requirement in analysis.requirements:
            checklist.items.append(
                ChecklistItem(
                    description=requirement,
                    success_criteria=f"'{requirement[:60]}' 요구가 산출물로 확인된다",
                )
            )

        # 제약은 상시 검증 항목으로 (가중치 낮음)
        for constraint in analysis.constraints:
            checklist.items.append(
                ChecklistItem(
                    description=f"제약 준수: {constraint}",
                    success_criteria="작업 전 과정에서 해당 제약을 위반하지 않는다",
                    weight=0.5,
                )
            )

        finalizer = _DOMAIN_FINALIZERS.get(analysis.domain)
        if finalizer is not None:
            checklist.items.append(
                ChecklistItem(
                    description=finalizer.description,
                    success_criteria=finalizer.success_criteria,
                    weight=1.5,
                )
            )

        if not checklist.items:
            checklist.items.append(
                ChecklistItem(
                    description=analysis.raw_request or "요청 수행",
                    success_criteria="요청에 대한 유효한 산출물이 생성된다",
                )
            )

        logger.info(
            "Checklist generated: id=%s domain=%s items=%d",
            checklist.checklist_id,
            checklist.domain,
            len(checklist.items),
        )
        return checklist
