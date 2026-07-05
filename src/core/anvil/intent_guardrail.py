"""의도 가드레일 - 진행 중 작업이 원래 의도에서 벗어나지 않았는지 자가 진단 (M4)."""

import logging
import re
from dataclasses import dataclass, field
from typing import List, Set

from src.core.anvil.request_analyzer import RequestAnalysis

logger = logging.getLogger(__name__)

# 토큰화에서 제외할 불용어 (한/영)
_STOPWORDS: Set[str] = {
    "그리고", "그래서", "하지만", "위해", "대한", "있는", "없는", "해줘", "해라", "합니다",
    "the", "a", "an", "and", "or", "but", "for", "to", "of", "in", "on", "with", "is", "are",
    "this", "that", "it", "be", "as", "at", "by", "from",
}

# 제약 문장에서 마커 자체는 내용 토큰이 아니므로 제외
_CONSTRAINT_MARKERS: Set[str] = {
    "금지", "하지", "마", "말", "것", "없이", "제외", "must", "not", "do", "without", "except",
}


def _tokenize(text: str) -> Set[str]:
    """의미 비교용 토큰 집합 추출 (2자 이상, 불용어 제거)."""
    tokens = re.findall(r"[\w가-힣]+", text.lower())
    return {t for t in tokens if len(t) >= 2 and t not in _STOPWORDS}


def _is_covered(token: str, token_set: Set[str]) -> bool:
    """토큰이 집합 내 토큰과 일치하거나 접두 관계면 반영된 것으로 본다.

    한국어 조사/어미 변화("수정" vs "수정하고")를 흡수하기 위한 처리.
    """
    if token in token_set:
        return True
    return any(
        t.startswith(token) or token.startswith(t) for t in token_set if len(t) >= 2
    )


def _overlap_count(targets: Set[str], observed: Set[str]) -> int:
    """targets 중 observed에 반영된 토큰 수."""
    return sum(1 for t in targets if _is_covered(t, observed))


@dataclass
class IntentAssessment:
    """의도 정렬 자가 진단 결과."""

    aligned: bool
    drift_score: float  # 0.0(완전 일치) ~ 1.0(완전 이탈)
    violated_constraints: List[str] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)


class IntentGuardrail:
    """원래 요청 분석 결과를 기준으로 현재 작업 상태의 의도 이탈을 주기적으로 진단한다.

    LLM 호출 없이 토큰 중첩 기반 휴리스틱으로 동작하므로 모든 단계에서
    저비용으로 호출할 수 있다. 이탈이 연속 감지되면 HITL 체크포인트를
    트리거하는 신호(`needs_human_review`)를 제공한다.
    """

    def __init__(
        self,
        analysis: RequestAnalysis,
        drift_threshold: float = 0.65,
        check_interval: int = 3,
        escalation_streak: int = 2,
    ):
        self.analysis = analysis
        self.drift_threshold = drift_threshold
        self.check_interval = max(1, check_interval)
        self.escalation_streak = max(1, escalation_streak)
        self.history: List[IntentAssessment] = []
        self._intent_tokens = _tokenize(analysis.raw_request) | {
            t for req in analysis.requirements for t in _tokenize(req)
        }

    def should_check(self, step_index: int) -> bool:
        """주기적 진단 시점 여부 (step_index는 1부터 시작하는 완료 단계 수)."""
        return step_index > 0 and step_index % self.check_interval == 0

    def evaluate(self, current_summary: str) -> IntentAssessment:
        """현재 작업 요약이 원래 의도와 정렬되어 있는지 평가."""
        reasons: List[str] = []
        summary_tokens = _tokenize(current_summary or "")

        if not summary_tokens or not self._intent_tokens:
            drift_score = 1.0 if self._intent_tokens else 0.0
            if drift_score:
                reasons.append("현재 작업 요약에서 원래 요청과 겹치는 내용을 찾지 못함")
        else:
            overlap = _overlap_count(self._intent_tokens, summary_tokens)
            drift_score = round(1.0 - overlap / len(self._intent_tokens), 4)
            if drift_score > self.drift_threshold:
                reasons.append(
                    f"원래 요청 핵심 토큰 {len(self._intent_tokens)}개 중 {overlap}개만 현재 작업에 반영됨"
                )

        violated = self._detect_constraint_violations(summary_tokens)
        if violated:
            reasons.append(f"제약 관련 활동 감지: {len(violated)}건")

        aligned = drift_score <= self.drift_threshold and not violated
        assessment = IntentAssessment(
            aligned=aligned,
            drift_score=drift_score,
            violated_constraints=violated,
            reasons=reasons,
        )
        self.history.append(assessment)
        logger.info(
            "Intent assessment: aligned=%s drift=%.2f violations=%d",
            aligned,
            drift_score,
            len(violated),
        )
        return assessment

    def needs_human_review(self) -> bool:
        """최근 진단이 연속으로 이탈이면 사람의 확인이 필요하다."""
        if len(self.history) < self.escalation_streak:
            return False
        recent = self.history[-self.escalation_streak:]
        return all(not a.aligned for a in recent)

    def _detect_constraint_violations(self, summary_tokens: Set[str]) -> List[str]:
        """제약 문장의 내용 토큰이 현재 작업에 등장하면 위반 후보로 표시."""
        violated: List[str] = []
        for constraint in self.analysis.constraints:
            content_tokens = _tokenize(constraint) - _CONSTRAINT_MARKERS
            if content_tokens and _overlap_count(content_tokens, summary_tokens):
                violated.append(constraint)
        return violated
