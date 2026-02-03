"""
Memory Learner

성공적인 컨텍스트 조합 패턴 학습, 실패한 컨텍스트 조합 회피,
사용자 피드백 기반 개선, A/B 테스트를 통한 최적화.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from collections import defaultdict
import json

logger = logging.getLogger(__name__)


@dataclass
class ContextCombination:
    """컨텍스트 조합."""
    context_types: List[str]
    token_allocation: Dict[str, int]
    priority_order: List[str]
    success_rate: float = 0.0
    usage_count: int = 0
    last_used: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LearningPattern:
    """학습 패턴."""
    pattern_id: str
    context_combination: ContextCombination
    success_cases: int = 0
    failure_cases: int = 0
    average_quality_score: float = 0.0
    learned_at: datetime = field(default_factory=datetime.now)


@dataclass
class UserFeedback:
    """사용자 피드백."""
    session_id: str
    feedback_type: str  # positive, negative, neutral
    feedback_text: Optional[str] = None
    quality_score: Optional[float] = None  # 0.0 ~ 1.0
    timestamp: datetime = field(default_factory=datetime.now)


class MemoryLearner:
    """
    메모리 학습 시스템.
    
    성공/실패 패턴 학습, 사용자 피드백 기반 개선,
    A/B 테스트를 통한 최적화를 담당합니다.
    """
    
    def __init__(self):
        """초기화."""
        self.successful_patterns: Dict[str, LearningPattern] = {}
        self.failed_patterns: Dict[str, LearningPattern] = {}
        self.user_feedbacks: List[UserFeedback] = []
        self.ab_tests: Dict[str, Dict[str, Any]] = {}
        
        logger.info("MemoryLearner initialized")
    
    def learn_from_session(
        self,
        session_id: str,
        context_combination: ContextCombination,
        success: bool,
        quality_score: Optional[float] = None
    ) -> bool:
        """
        세션에서 학습.
        
        Args:
            session_id: 세션 ID
            context_combination: 사용된 컨텍스트 조합
            success: 성공 여부
            quality_score: 품질 점수 (0.0 ~ 1.0)
            
        Returns:
            성공 여부
        """
        try:
            # 패턴 ID 생성
            pattern_id = self._generate_pattern_id(context_combination)
            
            if success:
                # 성공 패턴 학습
                if pattern_id in self.successful_patterns:
                    pattern = self.successful_patterns[pattern_id]
                    pattern.success_cases += 1
                    if quality_score:
                        # 이동 평균으로 품질 점수 업데이트
                        pattern.average_quality_score = (
                            pattern.average_quality_score * 0.7 + quality_score * 0.3
                        )
                else:
                    self.successful_patterns[pattern_id] = LearningPattern(
                        pattern_id=pattern_id,
                        context_combination=context_combination,
                        success_cases=1,
                        failure_cases=0,
                        average_quality_score=quality_score or 0.8
                    )
            else:
                # 실패 패턴 학습
                if pattern_id in self.failed_patterns:
                    pattern = self.failed_patterns[pattern_id]
                    pattern.failure_cases += 1
                else:
                    self.failed_patterns[pattern_id] = LearningPattern(
                        pattern_id=pattern_id,
                        context_combination=context_combination,
                        success_cases=0,
                        failure_cases=1,
                        average_quality_score=quality_score or 0.2
                    )
            
            # 컨텍스트 조합 사용 통계 업데이트
            context_combination.usage_count += 1
            context_combination.last_used = datetime.now()
            if success:
                context_combination.success_rate = (
                    context_combination.success_rate * 0.9 + 1.0 * 0.1
                )
            else:
                context_combination.success_rate = (
                    context_combination.success_rate * 0.9 + 0.0 * 0.1
                )
            
            logger.debug(f"Learned from session {session_id}: pattern={pattern_id}, success={success}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to learn from session {session_id}: {e}")
            return False
    
    def record_feedback(
        self,
        session_id: str,
        feedback_type: str,
        feedback_text: Optional[str] = None,
        quality_score: Optional[float] = None
    ) -> bool:
        """
        사용자 피드백 기록.
        
        Args:
            session_id: 세션 ID
            feedback_type: 피드백 타입 (positive, negative, neutral)
            feedback_text: 피드백 텍스트
            quality_score: 품질 점수
            
        Returns:
            성공 여부
        """
        try:
            feedback = UserFeedback(
                session_id=session_id,
                feedback_type=feedback_type,
                feedback_text=feedback_text,
                quality_score=quality_score,
                timestamp=datetime.now()
            )
            
            self.user_feedbacks.append(feedback)
            
            # 최근 1000개만 유지
            if len(self.user_feedbacks) > 1000:
                self.user_feedbacks = self.user_feedbacks[-1000:]
            
            logger.info(f"Feedback recorded: {session_id}, type={feedback_type}, score={quality_score}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to record feedback: {e}")
            return False
    
    def get_recommended_combination(
        self,
        context_types: List[str],
        available_tokens: int
    ) -> Optional[ContextCombination]:
        """
        추천 컨텍스트 조합 반환.
        
        Args:
            context_types: 필요한 컨텍스트 타입 목록
            available_tokens: 사용 가능한 토큰 수
            
        Returns:
            추천된 ContextCombination 또는 None
        """
        try:
            # 성공 패턴 중에서 유사한 것 찾기
            best_pattern = None
            best_score = 0.0
            
            for pattern_id, pattern in self.successful_patterns.items():
                combination = pattern.context_combination
                
                # 타입 일치도 계산
                type_match = len(set(context_types) & set(combination.context_types)) / max(
                    len(context_types), len(combination.context_types), 1
                )
                
                # 성공률과 품질 점수 고려
                success_score = pattern.average_quality_score * pattern.success_cases / (
                    pattern.success_cases + pattern.failure_cases + 1
                )
                
                # 종합 점수
                score = type_match * 0.4 + success_score * 0.6
                
                if score > best_score:
                    best_score = score
                    best_pattern = pattern
            
            if best_pattern:
                # 추천 조합 생성 (토큰 할당 조정)
                recommended = ContextCombination(
                    context_types=context_types,
                    token_allocation=self._calculate_token_allocation(
                        context_types, available_tokens, best_pattern.context_combination.token_allocation
                    ),
                    priority_order=best_pattern.context_combination.priority_order,
                    success_rate=best_pattern.context_combination.success_rate,
                    usage_count=best_pattern.context_combination.usage_count
                )
                return recommended
            
            # 추천 패턴이 없으면 기본 조합 반환
            return self._create_default_combination(context_types, available_tokens)
            
        except Exception as e:
            logger.error(f"Failed to get recommended combination: {e}")
            return None
    
    def avoid_failed_combinations(
        self,
        context_types: List[str]
    ) -> List[str]:
        """
        실패한 조합 회피.
        
        Args:
            context_types: 컨텍스트 타입 목록
            
        Returns:
            회피해야 할 타입 목록
        """
        avoid_types = []
        
        try:
            for pattern_id, pattern in self.failed_patterns.items():
                combination = pattern.context_combination
                
                # 실패율이 높은 조합의 타입들
                failure_rate = pattern.failure_cases / (pattern.success_cases + pattern.failure_cases + 1)
                if failure_rate > 0.7:  # 70% 이상 실패
                    for ctx_type in combination.context_types:
                        if ctx_type in context_types and ctx_type not in avoid_types:
                            avoid_types.append(ctx_type)
            
            return avoid_types
            
        except Exception as e:
            logger.error(f"Failed to get avoid combinations: {e}")
            return []
    
    def run_ab_test(
        self,
        test_id: str,
        variant_a: ContextCombination,
        variant_b: ContextCombination,
        sessions_per_variant: int = 10
    ) -> Dict[str, Any]:
        """
        A/B 테스트 실행.
        
        Args:
            test_id: 테스트 ID
            variant_a: 변형 A
            variant_b: 변형 B
            sessions_per_variant: 변형당 세션 수
            
        Returns:
            테스트 결과
        """
        try:
            test_result = {
                "test_id": test_id,
                "variant_a": {
                    "combination": variant_a,
                    "sessions": 0,
                    "successes": 0,
                    "average_quality": 0.0
                },
                "variant_b": {
                    "combination": variant_b,
                    "sessions": 0,
                    "successes": 0,
                    "average_quality": 0.0
                },
                "winner": None,
                "confidence": 0.0
            }
            
            self.ab_tests[test_id] = test_result
            
            logger.info(f"A/B test started: {test_id}")
            return test_result
            
        except Exception as e:
            logger.error(f"Failed to run A/B test: {e}")
            return {}
    
    def update_ab_test(
        self,
        test_id: str,
        variant: str,  # "a" or "b"
        success: bool,
        quality_score: Optional[float] = None
    ) -> bool:
        """
        A/B 테스트 결과 업데이트.
        
        Args:
            test_id: 테스트 ID
            variant: 변형 (a 또는 b)
            success: 성공 여부
            quality_score: 품질 점수
            
        Returns:
            성공 여부
        """
        try:
            if test_id not in self.ab_tests:
                logger.warning(f"A/B test not found: {test_id}")
                return False
            
            test_result = self.ab_tests[test_id]
            variant_key = f"variant_{variant}"
            
            if variant_key not in test_result:
                logger.error(f"Invalid variant: {variant}")
                return False
            
            variant_data = test_result[variant_key]
            variant_data["sessions"] += 1
            if success:
                variant_data["successes"] += 1
            
            if quality_score:
                variant_data["average_quality"] = (
                    variant_data["average_quality"] * 0.9 + quality_score * 0.1
                )
            
            # 승자 결정 (충분한 데이터가 있을 때)
            if (test_result["variant_a"]["sessions"] >= 10 and
                test_result["variant_b"]["sessions"] >= 10):
                a_score = test_result["variant_a"]["successes"] / test_result["variant_a"]["sessions"]
                b_score = test_result["variant_b"]["successes"] / test_result["variant_b"]["sessions"]
                
                if a_score > b_score * 1.1:  # 10% 이상 차이
                    test_result["winner"] = "a"
                    test_result["confidence"] = abs(a_score - b_score)
                elif b_score > a_score * 1.1:
                    test_result["winner"] = "b"
                    test_result["confidence"] = abs(b_score - a_score)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update A/B test: {e}")
            return False
    
    def _generate_pattern_id(self, combination: ContextCombination) -> str:
        """패턴 ID 생성."""
        types_str = "_".join(sorted(combination.context_types))
        return f"pattern_{types_str}"
    
    def _calculate_token_allocation(
        self,
        context_types: List[str],
        available_tokens: int,
        reference_allocation: Dict[str, int]
    ) -> Dict[str, int]:
        """토큰 할당 계산."""
        allocation = {}
        total_reference = sum(reference_allocation.values()) if reference_allocation else 1
        
        for ctx_type in context_types:
            if ctx_type in reference_allocation:
                ratio = reference_allocation[ctx_type] / total_reference
                allocation[ctx_type] = int(available_tokens * ratio)
            else:
                # 균등 분배
                allocation[ctx_type] = int(available_tokens / len(context_types))
        
        return allocation
    
    def _create_default_combination(
        self,
        context_types: List[str],
        available_tokens: int
    ) -> ContextCombination:
        """기본 조합 생성."""
        allocation = {}
        per_type = available_tokens // len(context_types) if context_types else available_tokens
        
        for ctx_type in context_types:
            allocation[ctx_type] = per_type
        
        return ContextCombination(
            context_types=context_types,
            token_allocation=allocation,
            priority_order=context_types
        )
    
    def get_learning_stats(self) -> Dict[str, Any]:
        """학습 통계."""
        return {
            "successful_patterns": len(self.successful_patterns),
            "failed_patterns": len(self.failed_patterns),
            "total_feedbacks": len(self.user_feedbacks),
            "active_ab_tests": len(self.ab_tests),
            "average_success_rate": sum(
                p.context_combination.success_rate
                for p in self.successful_patterns.values()
            ) / len(self.successful_patterns) if self.successful_patterns else 0.0
        }


# 전역 인스턴스
_memory_learner: Optional[MemoryLearner] = None


def get_memory_learner() -> MemoryLearner:
    """전역 메모리 학습자 인스턴스 반환."""
    global _memory_learner
    if _memory_learner is None:
        _memory_learner = MemoryLearner()
    return _memory_learner

