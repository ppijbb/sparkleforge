"""
Context Analytics

컨텍스트 사용 패턴 분석, 세션 복원 성공률 추적,
토큰 효율성 메트릭, 개인화 효과 측정.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict
import json

logger = logging.getLogger(__name__)


@dataclass
class ContextUsageMetric:
    """컨텍스트 사용 메트릭."""
    timestamp: datetime
    session_id: str
    total_tokens: int
    tokens_used: int
    chunks_count: int
    compression_applied: bool
    cache_hit: bool
    optimization_time_ms: float


@dataclass
class SessionRestoreMetric:
    """세션 복원 메트릭."""
    timestamp: datetime
    session_id: str
    restore_success: bool
    restore_time_ms: float
    chunks_restored: int
    memory_entries_restored: int


@dataclass
class PersonalizationMetric:
    """개인화 메트릭."""
    timestamp: datetime
    user_id: str
    session_id: str
    personalized_weights_used: bool
    cache_hit_rate: float
    quality_score: Optional[float] = None


class ContextAnalytics:
    """
    컨텍스트 분석 시스템.
    
    컨텍스트 사용 패턴, 세션 복원 성공률,
    토큰 효율성, 개인화 효과를 추적합니다.
    """
    
    def __init__(self):
        """초기화."""
        self.context_usage_metrics: List[ContextUsageMetric] = []
        self.session_restore_metrics: List[SessionRestoreMetric] = []
        self.personalization_metrics: List[PersonalizationMetric] = []
        
        logger.info("ContextAnalytics initialized")
    
    def record_context_usage(
        self,
        session_id: str,
        total_tokens: int,
        tokens_used: int,
        chunks_count: int,
        compression_applied: bool = False,
        cache_hit: bool = False,
        optimization_time_ms: float = 0.0
    ):
        """
        컨텍스트 사용 기록.
        
        Args:
            session_id: 세션 ID
            total_tokens: 전체 토큰 수
            tokens_used: 사용된 토큰 수
            chunks_count: 청크 수
            compression_applied: 압축 적용 여부
            cache_hit: 캐시 히트 여부
            optimization_time_ms: 최적화 시간 (ms)
        """
        try:
            metric = ContextUsageMetric(
                timestamp=datetime.now(),
                session_id=session_id,
                total_tokens=total_tokens,
                tokens_used=tokens_used,
                chunks_count=chunks_count,
                compression_applied=compression_applied,
                cache_hit=cache_hit,
                optimization_time_ms=optimization_time_ms
            )
            
            self.context_usage_metrics.append(metric)
            
            # 최근 1000개만 유지
            if len(self.context_usage_metrics) > 1000:
                self.context_usage_metrics = self.context_usage_metrics[-1000:]
            
            logger.debug(f"Context usage recorded: {session_id}, tokens={tokens_used}/{total_tokens}")
            
        except Exception as e:
            logger.error(f"Failed to record context usage: {e}")
    
    def record_session_restore(
        self,
        session_id: str,
        restore_success: bool,
        restore_time_ms: float,
        chunks_restored: int = 0,
        memory_entries_restored: int = 0
    ):
        """
        세션 복원 기록.
        
        Args:
            session_id: 세션 ID
            restore_success: 복원 성공 여부
            restore_time_ms: 복원 시간 (ms)
            chunks_restored: 복원된 청크 수
            memory_entries_restored: 복원된 메모리 항목 수
        """
        try:
            metric = SessionRestoreMetric(
                timestamp=datetime.now(),
                session_id=session_id,
                restore_success=restore_success,
                restore_time_ms=restore_time_ms,
                chunks_restored=chunks_restored,
                memory_entries_restored=memory_entries_restored
            )
            
            self.session_restore_metrics.append(metric)
            
            # 최근 1000개만 유지
            if len(self.session_restore_metrics) > 1000:
                self.session_restore_metrics = self.session_restore_metrics[-1000:]
            
            logger.debug(f"Session restore recorded: {session_id}, success={restore_success}, time={restore_time_ms}ms")
            
        except Exception as e:
            logger.error(f"Failed to record session restore: {e}")
    
    def record_personalization(
        self,
        user_id: str,
        session_id: str,
        personalized_weights_used: bool,
        cache_hit_rate: float = 0.0,
        quality_score: Optional[float] = None
    ):
        """
        개인화 효과 기록.
        
        Args:
            user_id: 사용자 ID
            session_id: 세션 ID
            personalized_weights_used: 개인화 가중치 사용 여부
            cache_hit_rate: 캐시 히트율
            quality_score: 품질 점수
        """
        try:
            metric = PersonalizationMetric(
                timestamp=datetime.now(),
                user_id=user_id,
                session_id=session_id,
                personalized_weights_used=personalized_weights_used,
                cache_hit_rate=cache_hit_rate,
                quality_score=quality_score
            )
            
            self.personalization_metrics.append(metric)
            
            # 최근 1000개만 유지
            if len(self.personalization_metrics) > 1000:
                self.personalization_metrics = self.personalization_metrics[-1000:]
            
            logger.debug(f"Personalization recorded: {user_id}, session={session_id}")
            
        except Exception as e:
            logger.error(f"Failed to record personalization: {e}")
    
    def get_context_usage_patterns(
        self,
        days: int = 7
    ) -> Dict[str, Any]:
        """
        컨텍스트 사용 패턴 분석.
        
        Args:
            days: 분석 기간 (일)
            
        Returns:
            사용 패턴 통계
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            recent_metrics = [
                m for m in self.context_usage_metrics
                if m.timestamp >= cutoff_date
            ]
            
            if not recent_metrics:
                return {"error": "No data available"}
            
            total_sessions = len(set(m.session_id for m in recent_metrics))
            total_tokens_used = sum(m.tokens_used for m in recent_metrics)
            total_tokens_available = sum(m.total_tokens for m in recent_metrics)
            avg_chunks = sum(m.chunks_count for m in recent_metrics) / len(recent_metrics)
            compression_rate = sum(1 for m in recent_metrics if m.compression_applied) / len(recent_metrics)
            cache_hit_rate = sum(1 for m in recent_metrics if m.cache_hit) / len(recent_metrics)
            avg_optimization_time = sum(m.optimization_time_ms for m in recent_metrics) / len(recent_metrics)
            
            token_efficiency = (total_tokens_used / total_tokens_available) * 100 if total_tokens_available > 0 else 0.0
            
            return {
                "period_days": days,
                "total_sessions": total_sessions,
                "total_optimizations": len(recent_metrics),
                "token_efficiency_percent": token_efficiency,
                "average_chunks_per_session": avg_chunks,
                "compression_rate": compression_rate,
                "cache_hit_rate": cache_hit_rate,
                "average_optimization_time_ms": avg_optimization_time
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze context usage patterns: {e}")
            return {"error": str(e)}
    
    def get_session_restore_stats(
        self,
        days: int = 7
    ) -> Dict[str, Any]:
        """
        세션 복원 통계.
        
        Args:
            days: 분석 기간 (일)
            
        Returns:
            복원 통계
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            recent_metrics = [
                m for m in self.session_restore_metrics
                if m.timestamp >= cutoff_date
            ]
            
            if not recent_metrics:
                return {"error": "No data available"}
            
            total_restores = len(recent_metrics)
            successful_restores = sum(1 for m in recent_metrics if m.restore_success)
            success_rate = successful_restores / total_restores if total_restores > 0 else 0.0
            avg_restore_time = sum(m.restore_time_ms for m in recent_metrics if m.restore_success) / successful_restores if successful_restores > 0 else 0.0
            avg_chunks_restored = sum(m.chunks_restored for m in recent_metrics if m.restore_success) / successful_restores if successful_restores > 0 else 0.0
            avg_memory_restored = sum(m.memory_entries_restored for m in recent_metrics if m.restore_success) / successful_restores if successful_restores > 0 else 0.0
            
            return {
                "period_days": days,
                "total_restores": total_restores,
                "successful_restores": successful_restores,
                "success_rate": success_rate,
                "average_restore_time_ms": avg_restore_time,
                "average_chunks_restored": avg_chunks_restored,
                "average_memory_entries_restored": avg_memory_restored
            }
            
        except Exception as e:
            logger.error(f"Failed to get session restore stats: {e}")
            return {"error": str(e)}
    
    def get_personalization_effectiveness(
        self,
        days: int = 7
    ) -> Dict[str, Any]:
        """
        개인화 효과 측정.
        
        Args:
            days: 분석 기간 (일)
            
        Returns:
            개인화 효과 통계
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            recent_metrics = [
                m for m in self.personalization_metrics
                if m.timestamp >= cutoff_date
            ]
            
            if not recent_metrics:
                return {"error": "No data available"}
            
            personalized_sessions = sum(1 for m in recent_metrics if m.personalized_weights_used)
            total_sessions = len(recent_metrics)
            personalization_rate = personalized_sessions / total_sessions if total_sessions > 0 else 0.0
            
            avg_cache_hit_rate = sum(m.cache_hit_rate for m in recent_metrics) / total_sessions if total_sessions > 0 else 0.0
            
            quality_scores = [m.quality_score for m in recent_metrics if m.quality_score is not None]
            avg_quality_score = sum(quality_scores) / len(quality_scores) if quality_scores else None
            
            return {
                "period_days": days,
                "total_sessions": total_sessions,
                "personalized_sessions": personalized_sessions,
                "personalization_rate": personalization_rate,
                "average_cache_hit_rate": avg_cache_hit_rate,
                "average_quality_score": avg_quality_score
            }
            
        except Exception as e:
            logger.error(f"Failed to get personalization effectiveness: {e}")
            return {"error": str(e)}
    
    def get_comprehensive_report(self) -> Dict[str, Any]:
        """종합 분석 보고서."""
        return {
            "context_usage": self.get_context_usage_patterns(),
            "session_restore": self.get_session_restore_stats(),
            "personalization": self.get_personalization_effectiveness(),
            "timestamp": datetime.now().isoformat()
        }


# 전역 인스턴스
_context_analytics: Optional[ContextAnalytics] = None


def get_context_analytics() -> ContextAnalytics:
    """전역 컨텍스트 분석 인스턴스 반환."""
    global _context_analytics
    if _context_analytics is None:
        _context_analytics = ContextAnalytics()
    return _context_analytics

