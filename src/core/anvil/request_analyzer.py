"""요청 분석기 - 오픈엔디드 요청의 요구사항 분석 및 문제 도메인 정의 (M3)."""

import logging
import re
import hashlib
import time
from dataclasses import dataclass, field
from typing import Dict, List

logger = logging.getLogger(__name__)

# 도메인별 시그널 키워드 (한/영)
_DOMAIN_SIGNALS: Dict[str, List[str]] = {
    "code": ["구현", "코드", "함수", "클래스", "리팩토링", "버그", "테스트", "implement", "code", "refactor", "bug", "test", "debug"],
    "file": ["파일", "디렉토리", "정리", "저장", "읽어", "써", "file", "directory", "organize", "write", "read"],
    "research": ["조사", "검색", "분석해", "리서치", "논문", "동향", "research", "search", "investigate", "paper", "trend"],
    "automation": ["자동화", "매일", "스케줄", "주기적", "반복", "automation", "schedule", "daily", "recurring", "cron"],
    "data": ["데이터", "통계", "차트", "시각화", "집계", "data", "statistics", "chart", "visualize", "aggregate"],
    "system": ["설치", "환경", "설정", "배포", "빌드", "install", "environment", "setup", "deploy", "build"],
}


@dataclass
class RequestAnalysis:
    """요청 분석 결과."""

    request_id: str = ""
    agent_identity: str = ""
    raw_request: str
    domain: str
    secondary_domains: List[str] = field(default_factory=list)
    requirements: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    confidence: float = 0.0


class RequestAnalyzer:
    """오픈엔디드 요청을 분석해 핵심 문제 도메인과 요구사항을 정의한다."""

    _MANDATE_PREFIX = "sparkleforge-anvil-mandate"

    def issue_mandate(self, request: str, agent_identity: str = "anvil-request-analyzer") -> str:
        """Issue a verifiable delegated mandate token for the given request."""
        digest = hashlib.sha256(f"{self._MANDATE_PREFIX}:{agent_identity}:{request}".encode("utf-8")).hexdigest()
        return f"{self._MANDATE_PREFIX}:{agent_identity}:{digest[:16]}"

    def analyze(self, request: str) -> RequestAnalysis:
        """요청 텍스트를 분석해 도메인·요구사항·제약을 추출."""
        if not request or not request.strip():
            return RequestAnalysis(raw_request=request or "", domain="general", confidence=0.0)

        text = request.strip()
        scores = self._score_domains(text)
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)

        domain, top_score = ranked[0] if ranked and ranked[0][1] > 0 else ("general", 0)
        secondary = [d for d, s in ranked[1:] if s > 0]
        total = sum(scores.values()) or 1
        confidence = min(top_score / total + 0.3, 1.0) if top_score else 0.3

        analysis = RequestAnalysis(
            raw_request=text,
            domain=domain,
            secondary_domains=secondary,
            requirements=self._extract_requirements(text),
            constraints=self._extract_constraints(text),
            confidence=round(confidence, 2),
        )
        logger.info(
            "Request analyzed: domain=%s secondary=%s requirements=%d confidence=%.2f",
            analysis.domain,
            analysis.secondary_domains,
            len(analysis.requirements),
            analysis.confidence,
        )
        return analysis

    @staticmethod
    def _score_domains(text: str) -> Dict[str, int]:
        lowered = text.lower()
        return {
            domain: sum(1 for kw in keywords if kw in lowered)
            for domain, keywords in _DOMAIN_SIGNALS.items()
        }

    @staticmethod
    def _extract_requirements(text: str) -> List[str]:
        """문장 단위로 실행 가능한 요구사항 후보를 분리."""
        parts = re.split(r"[.\n;·]|(?:\d+\))", text)
        return [p.strip() for p in parts if len(p.strip()) >= 5]

    @staticmethod
    def _extract_constraints(text: str) -> List[str]:
        """금지/제한 표현을 제약으로 수집."""
        constraint_markers = ["금지", "하지 마", "말 것", "없이", "제외", "must not", "do not", "without", "except"]
        return [
            sentence.strip()
            for sentence in re.split(r"[.\n]", text)
            if any(marker in sentence.lower() for marker in constraint_markers) and sentence.strip()
        ]
