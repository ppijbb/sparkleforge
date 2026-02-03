"""
Council Activator - 자동 활성화 로직

다방면 검토가 필요한 경우 자동으로 Council을 활성화.
바이브 코딩 선서 준수: 하드 코딩 제거, 실제 복잡도 분석 사용.
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from src.core.researcher_config import get_council_config, get_agent_config

logger = logging.getLogger(__name__)


@dataclass
class ActivationDecision:
    """활성화 결정 결과."""
    should_activate: bool
    reason: str
    confidence: float  # 0.0-1.0


class CouncilActivator:
    """Council 자동 활성화 판단 클래스."""
    
    def __init__(self):
        """초기화."""
        self.council_config = get_council_config()
        self.agent_config = get_agent_config()
    
    def should_activate(
        self,
        process_type: str,
        query: Optional[str] = None,
        complexity_score: Optional[float] = None,
        context: Optional[Dict[str, Any]] = None,
        force_activate: bool = False
    ) -> ActivationDecision:
        """
        Council 활성화 여부 판단.
        
        Args:
            process_type: 프로세스 타입 ('planning', 'execution', 'evaluation', 'verification', 'synthesis')
            query: 사용자 쿼리 (복잡도 분석용)
            complexity_score: 복잡도 점수 (0.0-1.0, None이면 자동 계산)
            context: 추가 컨텍스트 (도메인, 위험도 등)
            force_activate: 강제 활성화 플래그
        
        Returns:
            ActivationDecision 객체
        """
        # 강제 활성화
        if force_activate:
            return ActivationDecision(
                should_activate=True,
                reason="User explicitly requested council activation",
                confidence=1.0
            )
        
        # Council이 비활성화된 경우
        if not self.council_config.enabled:
            return ActivationDecision(
                should_activate=False,
                reason="Council is disabled in configuration",
                confidence=1.0
            )
        
        # 프로세스별 활성화 설정 확인
        process_enabled = self._is_process_enabled(process_type)
        if not process_enabled:
            return ActivationDecision(
                should_activate=False,
                reason=f"Council is disabled for {process_type} process",
                confidence=1.0
            )
        
        # 자동 활성화가 비활성화된 경우
        if not self.council_config.auto_activate:
            return ActivationDecision(
                should_activate=False,
                reason="Auto-activation is disabled in configuration",
                confidence=1.0
            )
        
        # 복잡도 점수 계산 (제공되지 않은 경우)
        if complexity_score is None:
            complexity_score = self._calculate_complexity(query, context)
        
        # 복잡도 기반 판단
        if complexity_score >= self.council_config.min_complexity_for_auto:
            return ActivationDecision(
                should_activate=True,
                reason=f"Complexity score {complexity_score:.2f} exceeds threshold {self.council_config.min_complexity_for_auto}",
                confidence=min(complexity_score, 1.0)
            )
        
        # 다방면 검토 필요성 판단
        multi_facet_needed = self._needs_multi_facet_review(query, context)
        if multi_facet_needed:
            return ActivationDecision(
                should_activate=True,
                reason="Multi-facet review is needed",
                confidence=0.8
            )
        
        # 위험도 기반 판단
        risk_level = self._assess_risk_level(query, context)
        if risk_level > 0.7:
            return ActivationDecision(
                should_activate=True,
                reason=f"High risk level detected: {risk_level:.2f}",
                confidence=risk_level
            )
        
        # 기본값: 비활성화
        return ActivationDecision(
            should_activate=False,
            reason="No activation criteria met",
            confidence=0.5
        )
    
    def _is_process_enabled(self, process_type: str) -> bool:
        """프로세스 타입별 활성화 설정 확인."""
        process_map = {
            'planning': self.council_config.enable_for_planning,
            'execution': self.council_config.enable_for_execution,
            'evaluation': self.council_config.enable_for_evaluation,
            'verification': self.council_config.enable_for_verification,
            'synthesis': self.council_config.enable_for_synthesis,
        }
        return process_map.get(process_type, False)
    
    def _calculate_complexity(self, query: Optional[str], context: Optional[Dict[str, Any]]) -> float:
        """
        복잡도 점수 계산 (0.0-1.0).
        
        Args:
            query: 사용자 쿼리
            context: 추가 컨텍스트
        
        Returns:
            복잡도 점수
        """
        if query is None:
            return 0.5  # 기본값
        
        complexity = 0.0
        
        # 쿼리 길이 기반 복잡도
        query_length = len(query.split())
        if query_length > 50:
            complexity += 0.3
        elif query_length > 20:
            complexity += 0.2
        else:
            complexity += 0.1
        
        # 복잡한 키워드 감지
        complex_keywords = [
            'analyze', 'compare', 'evaluate', 'synthesize', 'comprehensive',
            'multiple', 'various', 'different', 'complex', 'detailed',
            '분석', '비교', '평가', '종합', '포괄적', '다양한', '복잡한'
        ]
        query_lower = query.lower()
        keyword_count = sum(1 for keyword in complex_keywords if keyword in query_lower)
        complexity += min(keyword_count * 0.1, 0.3)
        
        # 컨텍스트 기반 복잡도
        if context:
            # 다중 도메인 감지
            if context.get('domains') and len(context.get('domains', [])) > 1:
                complexity += 0.2
            
            # 다중 단계 감지
            if context.get('steps') and len(context.get('steps', [])) > 3:
                complexity += 0.2
        
        return min(complexity, 1.0)
    
    def _needs_multi_facet_review(self, query: Optional[str], context: Optional[Dict[str, Any]]) -> bool:
        """
        다방면 검토 필요성 판단.
        
        Args:
            query: 사용자 쿼리
            context: 추가 컨텍스트
        
        Returns:
            다방면 검토 필요 여부
        """
        if query is None:
            return False
        
        # 다방면 검토 키워드
        multi_facet_keywords = [
            'perspective', 'viewpoint', 'opinion', 'debate', 'controversy',
            'disagreement', 'conflicting', 'different views',
            '관점', '견해', '논쟁', '의견', '상반된', '다른 시각'
        ]
        
        query_lower = query.lower()
        has_multi_facet_keywords = any(keyword in query_lower for keyword in multi_facet_keywords)
        
        # 컨텍스트에서 다방면 검토 필요성 확인
        if context:
            if context.get('requires_multi_perspective', False):
                return True
            if context.get('controversial_topic', False):
                return True
        
        return has_multi_facet_keywords
    
    def _assess_risk_level(self, query: Optional[str], context: Optional[Dict[str, Any]]) -> float:
        """
        위험도 평가 (0.0-1.0).
        
        Args:
            query: 사용자 쿼리
            context: 추가 컨텍스트
        
        Returns:
            위험도 점수
        """
        risk = 0.0
        
        if query is None:
            return risk
        
        # 고위험 키워드
        high_risk_keywords = [
            'critical', 'important', 'decision', 'conclusion', 'final',
            'medical', 'legal', 'financial', 'safety', 'security',
            '중요한', '결정', '결론', '최종', '의료', '법률', '금융', '안전'
        ]
        
        query_lower = query.lower()
        risk_keyword_count = sum(1 for keyword in high_risk_keywords if keyword in query_lower)
        risk += min(risk_keyword_count * 0.15, 0.6)
        
        # 컨텍스트 기반 위험도
        if context:
            if context.get('high_stakes', False):
                risk += 0.3
            if context.get('requires_accuracy', False):
                risk += 0.2
        
        return min(risk, 1.0)


# 전역 인스턴스
_activator_instance: Optional[CouncilActivator] = None


def get_council_activator() -> CouncilActivator:
    """Council Activator 싱글톤 인스턴스 반환."""
    global _activator_instance
    if _activator_instance is None:
        _activator_instance = CouncilActivator()
    return _activator_instance

