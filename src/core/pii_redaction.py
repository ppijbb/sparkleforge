"""
PII Redaction

백서 요구사항: 세션 데이터 저장 전 PII 자동 감지 및 제거
"""

import logging
import re
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PIIMatch:
    """PII 매치 정보."""
    pii_type: str
    value: str
    start_pos: int
    end_pos: int
    confidence: float  # 0.0 ~ 1.0


# PII 패턴 정의
PII_PATTERNS = {
    "email": re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
    "phone": re.compile(r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'),
    "ssn": re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
    "credit_card": re.compile(r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b'),
    "ip_address": re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b'),
    "mac_address": re.compile(r'\b([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2})\b'),
    "url": re.compile(r'https?://[^\s<>"{}|\\^`\[\]]+'),
    "date_of_birth": re.compile(r'\b(?:0[1-9]|1[0-2])[/-](?:0[1-9]|[12]\d|3[01])[/-](?:19|20)\d{2}\b'),
}


class PIIRedactor:
    """
    PII 제거 시스템.
    
    백서 요구사항: 세션 데이터 저장 전 PII 자동 감지 및 제거
    """
    
    def __init__(self, use_llm_detection: bool = False):
        """
        초기화.
        
        Args:
            use_llm_detection: LLM 기반 PII 감지 사용 여부 (더 정확하지만 느림)
        """
        self.use_llm_detection = use_llm_detection
        self.patterns = PII_PATTERNS
        logger.info(f"PIIRedactor initialized (LLM detection: {use_llm_detection})")
    
    def detect_pii(self, text: str) -> List[PIIMatch]:
        """
        텍스트에서 PII 감지.
        
        Args:
            text: 검사할 텍스트
            
        Returns:
            PII 매치 목록
        """
        matches = []
        
        # 패턴 기반 감지
        for pii_type, pattern in self.patterns.items():
            for match in pattern.finditer(text):
                matches.append(PIIMatch(
                    pii_type=pii_type,
                    value=match.group(),
                    start_pos=match.start(),
                    end_pos=match.end(),
                    confidence=0.9  # 패턴 매치는 높은 신뢰도
                ))
        
        # 중복 제거 (겹치는 매치)
        matches = self._deduplicate_matches(matches)
        
        # LLM 기반 감지 (선택적)
        if self.use_llm_detection:
            llm_matches = self._detect_pii_with_llm(text)
            matches.extend(llm_matches)
            matches = self._deduplicate_matches(matches)
        
        return sorted(matches, key=lambda m: m.start_pos)
    
    def redact_text(self, text: str, replacement: str = "[REDACTED]") -> Tuple[str, List[PIIMatch]]:
        """
        텍스트에서 PII 제거.
        
        Args:
            text: 원본 텍스트
            replacement: 대체 문자열
            
        Returns:
            (제거된 텍스트, 감지된 PII 목록)
        """
        matches = self.detect_pii(text)
        
        if not matches:
            return text, []
        
        # 역순으로 제거 (인덱스 변경 방지)
        redacted_text = text
        for match in reversed(matches):
            redacted_text = (
                redacted_text[:match.start_pos] +
                f"[{match.pii_type.upper()}_REDACTED]" +
                redacted_text[match.end_pos:]
            )
        
        logger.info(f"Redacted {len(matches)} PII matches from text")
        return redacted_text, matches
    
    def redact_session_data(self, session_data: Dict[str, Any]) -> Tuple[Dict[str, Any], List[PIIMatch]]:
        """
        세션 데이터에서 PII 제거.
        
        Args:
            session_data: 세션 데이터
            
        Returns:
            (제거된 세션 데이터, 감지된 PII 목록)
        """
        all_matches = []
        redacted_data = {}
        
        for key, value in session_data.items():
            if isinstance(value, str):
                redacted_value, matches = self.redact_text(value)
                redacted_data[key] = redacted_value
                all_matches.extend(matches)
            elif isinstance(value, dict):
                redacted_value, matches = self.redact_session_data(value)
                redacted_data[key] = redacted_value
                all_matches.extend(matches)
            elif isinstance(value, list):
                redacted_list = []
                for item in value:
                    if isinstance(item, str):
                        redacted_item, matches = self.redact_text(item)
                        redacted_list.append(redacted_item)
                        all_matches.extend(matches)
                    elif isinstance(item, dict):
                        redacted_item, matches = self.redact_session_data(item)
                        redacted_list.append(redacted_item)
                        all_matches.extend(matches)
                    else:
                        redacted_list.append(item)
                redacted_data[key] = redacted_list
            else:
                redacted_data[key] = value
        
        return redacted_data, all_matches
    
    def _deduplicate_matches(self, matches: List[PIIMatch]) -> List[PIIMatch]:
        """중복 매치 제거."""
        if not matches:
            return []
        
        # 시작 위치 기준 정렬
        sorted_matches = sorted(matches, key=lambda m: (m.start_pos, -m.end_pos))
        deduplicated = [sorted_matches[0]]
        
        for match in sorted_matches[1:]:
            # 이전 매치와 겹치지 않으면 추가
            last_match = deduplicated[-1]
            if match.start_pos >= last_match.end_pos:
                deduplicated.append(match)
            elif match.confidence > last_match.confidence:
                # 더 높은 신뢰도로 교체
                deduplicated[-1] = match
        
        return deduplicated
    
    def _detect_pii_with_llm(self, text: str) -> List[PIIMatch]:
        """LLM을 사용한 PII 감지 (더 정확하지만 느림)."""
        # TODO: LLM 기반 PII 감지 구현
        # Issue: High priority - LLM 기반 PII 감지 기능 구현 필요
        # 현재는 패턴 기반만 사용
        # 향후 구현 시 LLM을 사용하여 더 정확한 PII 감지 가능
        return []


# 전역 인스턴스
_pii_redactor: Optional[PIIRedactor] = None


def get_pii_redactor(use_llm_detection: bool = False) -> PIIRedactor:
    """전역 PII 제거기 인스턴스 반환."""
    global _pii_redactor
    if _pii_redactor is None:
        _pii_redactor = PIIRedactor(use_llm_detection=use_llm_detection)
    return _pii_redactor

