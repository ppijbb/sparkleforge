#!/usr/bin/env python3
"""
Research Recommender for Local Researcher Project (v2.0 - 8대 혁신)

Production-grade 연구 추천 시스템.
현재 주제와 유사한 과거 연구를 찾고,
벡터 유사도 기반 추천을 통해 사용자에게
관련 연구를 제안합니다.

2025년 10월 최신 기술 스택:
- Vector similarity search
- Time-weighted recommendations
- User preference integration
- Production-grade recommendation engine
"""

import asyncio
import logging
import json
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
from collections import defaultdict
import math

# Import dependencies
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.storage.hybrid_storage import HybridStorage, ResearchMemory
from src.learning.user_profiler import UserProfiler, UserProfile
from src.core.logging_config import get_logger

logger = get_logger(__name__)


class RecommendationType(Enum):
    """추천 타입."""
    SIMILAR_RESEARCH = "similar_research"
    RELATED_TOPICS = "related_topics"
    CONTINUATION = "continuation"
    CROSS_DOMAIN = "cross_domain"
    TRENDING = "trending"


@dataclass
class Recommendation:
    """추천 결과."""
    recommendation_id: str
    type: RecommendationType
    title: str
    description: str
    similarity_score: float
    confidence: float
    research_id: Optional[str] = None
    topic: Optional[str] = None
    metadata: Dict[str, Any] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)
        if self.metadata is None:
            self.metadata = {}


@dataclass
class RecommendationContext:
    """추천 컨텍스트."""
    user_id: str
    current_topic: str
    research_depth: str
    preferred_sources: List[str]
    domain_expertise: List[str]
    recent_topics: List[str]
    time_preference: str = "recent"  # recent, all_time, balanced


class ResearchRecommender:
    """
    Production-grade 연구 추천 시스템.
    
    Features:
    - 현재 주제와 유사한 과거 연구 찾기
    - 벡터 유사도 기반 추천
    - 시간 가중치 적용 (최근 연구 우선)
    - 사용자 선호도 통합
    - 다양한 추천 타입 지원
    """
    
    def __init__(self, storage: HybridStorage, profiler: UserProfiler):
        """
        연구 추천 시스템 초기화.
        
        Args:
            storage: 하이브리드 스토리지
            profiler: 사용자 프로파일러
        """
        self.storage = storage
        self.profiler = profiler
        
        # 추천 설정
        self.config = {
            'max_recommendations': 10,
            'min_similarity_threshold': 0.3,
            'time_decay_factor': 0.1,  # 시간에 따른 가중치 감소
            'cross_domain_threshold': 0.5,
            'trending_window_days': 30
        }
        
        # 추천 캐시
        self.recommendation_cache = {}
        self.cache_ttl_hours = 6
        
        logger.info("ResearchRecommender initialized")
    
    async def get_recommendations(
        self,
        context: RecommendationContext,
        recommendation_types: Optional[List[RecommendationType]] = None
    ) -> List[Recommendation]:
        """
        추천을 생성합니다.
        
        Args:
            context: 추천 컨텍스트
            recommendation_types: 추천 타입 (선택사항)
            
        Returns:
            List[Recommendation]: 추천 결과
        """
        try:
            if recommendation_types is None:
                recommendation_types = [
                    RecommendationType.SIMILAR_RESEARCH,
                    RecommendationType.RELATED_TOPICS,
                    RecommendationType.CONTINUATION
                ]
            
            # 캐시 확인
            cache_key = self._generate_cache_key(context, recommendation_types)
            if cache_key in self.recommendation_cache:
                cached_data = self.recommendation_cache[cache_key]
                if self._is_cache_valid(cached_data['timestamp']):
                    logger.debug(f"Cache hit for recommendations: {cache_key}")
                    return cached_data['recommendations']
            
            # 추천 생성
            all_recommendations = []
            
            for rec_type in recommendation_types:
                recommendations = await self._generate_recommendations_by_type(
                    context, rec_type
                )
                all_recommendations.extend(recommendations)
            
            # 추천 정렬 및 필터링
            filtered_recommendations = self._filter_and_rank_recommendations(
                all_recommendations, context
            )
            
            # 캐시 저장
            self.recommendation_cache[cache_key] = {
                'recommendations': filtered_recommendations,
                'timestamp': datetime.now(timezone.utc)
            }
            
            logger.info(f"Generated {len(filtered_recommendations)} recommendations for user {context.user_id}")
            return filtered_recommendations
            
        except Exception as e:
            logger.error(f"Failed to get recommendations: {e}")
            return []
    
    async def _generate_recommendations_by_type(
        self,
        context: RecommendationContext,
        rec_type: RecommendationType
    ) -> List[Recommendation]:
        """타입별 추천을 생성합니다."""
        try:
            if rec_type == RecommendationType.SIMILAR_RESEARCH:
                return await self._generate_similar_research_recommendations(context)
            elif rec_type == RecommendationType.RELATED_TOPICS:
                return await self._generate_related_topics_recommendations(context)
            elif rec_type == RecommendationType.CONTINUATION:
                return await self._generate_continuation_recommendations(context)
            elif rec_type == RecommendationType.CROSS_DOMAIN:
                return await self._generate_cross_domain_recommendations(context)
            elif rec_type == RecommendationType.TRENDING:
                return await self._generate_trending_recommendations(context)
            else:
                return []
                
        except Exception as e:
            logger.error(f"Failed to generate {rec_type.value} recommendations: {e}")
            return []
    
    async def _generate_similar_research_recommendations(
        self,
        context: RecommendationContext
    ) -> List[Recommendation]:
        """유사한 연구 추천을 생성합니다."""
        try:
            # 유사한 연구 검색
            similar_research = await self.storage.search_similar_research(
                query=context.current_topic,
                user_id=context.user_id,
                limit=self.config['max_recommendations'],
                similarity_threshold=self.config['min_similarity_threshold']
            )
            
            recommendations = []
            for research in similar_research:
                # 시간 가중치 적용
                time_weight = self._calculate_time_weight(research.timestamp, context.time_preference)
                final_score = research.similarity_score * time_weight
                
                if final_score >= self.config['min_similarity_threshold']:
                    recommendation = Recommendation(
                        recommendation_id=f"similar_{research.research_id}",
                        type=RecommendationType.SIMILAR_RESEARCH,
                        title=f"Similar Research: {research.summary[:100]}...",
                        description=f"Research on {research.metadata.get('topic', 'unknown topic')} with {final_score:.2f} similarity",
                        similarity_score=final_score,
                        confidence=min(1.0, final_score * 1.2),
                        research_id=research.research_id,
                        topic=research.metadata.get('topic', ''),
                        metadata={
                            'original_similarity': research.similarity_score,
                            'time_weight': time_weight,
                            'timestamp': research.timestamp.isoformat()
                        }
                    )
                    recommendations.append(recommendation)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Failed to generate similar research recommendations: {e}")
            return []
    
    async def _generate_related_topics_recommendations(
        self,
        context: RecommendationContext
    ) -> List[Recommendation]:
        """관련 주제 추천을 생성합니다."""
        try:
            # 사용자 프로파일 가져오기
            profile = await self.profiler.get_or_create_profile(context.user_id)
            
            # 관련 주제 찾기
            related_topics = []
            
            # 사용자의 과거 주제에서 관련성 높은 것들 찾기
            for topic in profile.preferred_topics:
                if topic != context.current_topic:
                    # 간단한 키워드 기반 관련성 계산
                    relatedness = self._calculate_topic_relatedness(
                        context.current_topic, topic
                    )
                    if relatedness > 0.3:
                        related_topics.append((topic, relatedness))
            
            # 도메인 전문성 기반 주제 제안
            for domain in profile.domain_expertise:
                if domain not in context.current_topic.lower():
                    related_topics.append((f"Research in {domain}", 0.4))
            
            recommendations = []
            for topic, relatedness in related_topics[:5]:  # 상위 5개
                recommendation = Recommendation(
                    recommendation_id=f"topic_{hash(topic)}",
                    type=RecommendationType.RELATED_TOPICS,
                    title=f"Related Topic: {topic}",
                    description=f"Explore research in {topic} based on your interests",
                    similarity_score=relatedness,
                    confidence=min(1.0, relatedness * 1.5),
                    topic=topic,
                    metadata={
                        'relatedness_score': relatedness,
                        'source': 'user_profile'
                    }
                )
                recommendations.append(recommendation)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Failed to generate related topics recommendations: {e}")
            return []
    
    async def _generate_continuation_recommendations(
        self,
        context: RecommendationContext
    ) -> List[Recommendation]:
        """연속 연구 추천을 생성합니다."""
        try:
            # 사용자의 최근 연구 히스토리 가져오기
            recent_research = await self.storage.get_user_research_history(
                context.user_id, limit=10
            )
            
            # 현재 주제와 연관된 미완성 연구 찾기
            continuation_candidates = []
            
            for research in recent_research:
                if research.topic != context.current_topic:
                    # 연속성 점수 계산
                    continuity_score = self._calculate_continuity_score(
                        context.current_topic, research
                    )
                    
                    if continuity_score > 0.4:
                        continuation_candidates.append((research, continuity_score))
            
            # 점수순으로 정렬
            continuation_candidates.sort(key=lambda x: x[1], reverse=True)
            
            recommendations = []
            for research, score in continuation_candidates[:3]:  # 상위 3개
                recommendation = Recommendation(
                    recommendation_id=f"continuation_{research.research_id}",
                    type=RecommendationType.CONTINUATION,
                    title=f"Continue Research: {research.topic}",
                    description=f"Build upon your previous research on {research.topic}",
                    similarity_score=score,
                    confidence=min(1.0, score * 1.3),
                    research_id=research.research_id,
                    topic=research.topic,
                    metadata={
                        'continuity_score': score,
                        'previous_research_date': research.timestamp.isoformat(),
                        'previous_summary': research.summary
                    }
                )
                recommendations.append(recommendation)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Failed to generate continuation recommendations: {e}")
            return []
    
    async def _generate_cross_domain_recommendations(
        self,
        context: RecommendationContext
    ) -> List[Recommendation]:
        """교차 도메인 추천을 생성합니다."""
        try:
            # 현재 주제와 다른 도메인의 연구 찾기
            profile = await self.profiler.get_or_create_profile(context.user_id)
            
            # 사용자의 도메인 전문성과 다른 영역 찾기
            cross_domain_topics = []
            
            # 일반적인 교차 도메인 조합
            cross_domain_mapping = {
                'technology': ['business', 'ethics', 'society'],
                'science': ['technology', 'medicine', 'environment'],
                'business': ['technology', 'psychology', 'economics'],
                'medicine': ['technology', 'ethics', 'society'],
                'art': ['technology', 'psychology', 'society']
            }
            
            current_domain = self._extract_domain(context.current_topic)
            if current_domain in cross_domain_mapping:
                for related_domain in cross_domain_mapping[current_domain]:
                    if related_domain not in profile.domain_expertise:
                        cross_domain_topics.append(related_domain)
            
            recommendations = []
            for domain in cross_domain_topics[:3]:
                recommendation = Recommendation(
                    recommendation_id=f"cross_domain_{domain}",
                    type=RecommendationType.CROSS_DOMAIN,
                    title=f"Cross-Domain Research: {domain.title()}",
                    description=f"Explore {domain} applications of your current research",
                    similarity_score=self.config['cross_domain_threshold'],
                    confidence=0.6,
                    topic=f"{context.current_topic} in {domain}",
                    metadata={
                        'cross_domain': domain,
                        'original_topic': context.current_topic
                    }
                )
                recommendations.append(recommendation)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Failed to generate cross-domain recommendations: {e}")
            return []
    
    async def _generate_trending_recommendations(
        self,
        context: RecommendationContext
    ) -> List[Recommendation]:
        """트렌딩 추천을 생성합니다."""
        try:
            # 최근 연구에서 인기 있는 주제 찾기
            trending_window = datetime.now(timezone.utc) - timedelta(
                days=self.config['trending_window_days']
            )
            
            # 모든 사용자의 최근 연구 가져오기
            all_recent_research = []
            
            # 실제로는 모든 사용자의 연구를 가져와야 하지만,
            # 여기서는 현재 사용자의 최근 연구를 사용
            recent_research = await self.storage.get_user_research_history(
                context.user_id, limit=50
            )
            
            # 최근 연구 필터링
            recent_research = [
                r for r in recent_research 
                if r.timestamp >= trending_window
            ]
            
            # 주제별 빈도 계산
            topic_counts = defaultdict(int)
            for research in recent_research:
                topic_counts[research.topic] += 1
            
            # 트렌딩 주제 찾기
            trending_topics = sorted(
                topic_counts.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:5]
            
            recommendations = []
            for topic, count in trending_topics:
                if topic != context.current_topic:
                    recommendation = Recommendation(
                        recommendation_id=f"trending_{hash(topic)}",
                        type=RecommendationType.TRENDING,
                        title=f"Trending Topic: {topic}",
                        description=f"Popular research topic with {count} recent studies",
                        similarity_score=0.5,  # 트렌딩은 고정 점수
                        confidence=min(1.0, count * 0.1),
                        topic=topic,
                        metadata={
                            'trending_count': count,
                            'trending_window_days': self.config['trending_window_days']
                        }
                    )
                    recommendations.append(recommendation)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Failed to generate trending recommendations: {e}")
            return []
    
    def _calculate_time_weight(
        self,
        timestamp: datetime,
        time_preference: str
    ) -> float:
        """시간 가중치를 계산합니다."""
        try:
            now = datetime.now(timezone.utc)
            days_old = (now - timestamp).days
            
            if time_preference == "recent":
                # 최근일수록 높은 가중치
                return max(0.1, 1.0 - (days_old * self.config['time_decay_factor']))
            elif time_preference == "all_time":
                # 시간 무관
                return 1.0
            else:  # balanced
                # 균형잡힌 가중치
                return max(0.3, 1.0 - (days_old * self.config['time_decay_factor'] * 0.5))
                
        except Exception as e:
            logger.error(f"Failed to calculate time weight: {e}")
            return 1.0
    
    def _calculate_topic_relatedness(self, topic1: str, topic2: str) -> float:
        """주제 간 관련성을 계산합니다."""
        try:
            # 간단한 키워드 기반 관련성 계산
            words1 = set(topic1.lower().split())
            words2 = set(topic2.lower().split())
            
            if not words1 or not words2:
                return 0.0
            
            # Jaccard 유사도
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            
            return intersection / union if union > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Failed to calculate topic relatedness: {e}")
            return 0.0
    
    def _calculate_continuity_score(
        self,
        current_topic: str,
        previous_research: ResearchMemory
    ) -> float:
        """연속성 점수를 계산합니다."""
        try:
            # 주제 관련성
            topic_relatedness = self._calculate_topic_relatedness(
                current_topic, previous_research.topic
            )
            
            # 시간 가중치 (최근일수록 높음)
            days_since = (datetime.now(timezone.utc) - previous_research.timestamp).days
            time_weight = max(0.1, 1.0 - (days_since / 365))  # 1년 기준
            
            # 신뢰도 점수
            confidence_weight = previous_research.confidence_score
            
            # 종합 점수
            continuity_score = (
                topic_relatedness * 0.5 +
                time_weight * 0.3 +
                confidence_weight * 0.2
            )
            
            return min(1.0, continuity_score)
            
        except Exception as e:
            logger.error(f"Failed to calculate continuity score: {e}")
            return 0.0
    
    def _extract_domain(self, topic: str) -> str:
        """주제에서 도메인을 추출합니다."""
        topic_lower = topic.lower()
        
        domain_keywords = {
            'technology': ['tech', 'ai', 'software', 'computer', 'digital', 'cyber'],
            'science': ['science', 'research', 'study', 'experiment', 'analysis'],
            'business': ['business', 'market', 'company', 'finance', 'economy'],
            'medicine': ['medical', 'health', 'medicine', 'clinical', 'patient'],
            'art': ['art', 'design', 'creative', 'culture', 'aesthetic']
        }
        
        for domain, keywords in domain_keywords.items():
            if any(keyword in topic_lower for keyword in keywords):
                return domain
        
        return 'general'
    
    def _filter_and_rank_recommendations(
        self,
        recommendations: List[Recommendation],
        context: RecommendationContext
    ) -> List[Recommendation]:
        """추천을 필터링하고 순위를 매깁니다."""
        try:
            # 중복 제거
            unique_recommendations = {}
            for rec in recommendations:
                key = f"{rec.type.value}_{rec.topic or rec.research_id}"
                if key not in unique_recommendations or rec.similarity_score > unique_recommendations[key].similarity_score:
                    unique_recommendations[key] = rec
            
            filtered_recommendations = list(unique_recommendations.values())
            
            # 최소 임계값 필터링
            filtered_recommendations = [
                rec for rec in filtered_recommendations
                if rec.similarity_score >= self.config['min_similarity_threshold']
            ]
            
            # 점수순 정렬
            filtered_recommendations.sort(
                key=lambda x: (x.similarity_score * x.confidence),
                reverse=True
            )
            
            # 최대 개수 제한
            return filtered_recommendations[:self.config['max_recommendations']]
            
        except Exception as e:
            logger.error(f"Failed to filter and rank recommendations: {e}")
            return recommendations
    
    def _generate_cache_key(
        self,
        context: RecommendationContext,
        recommendation_types: List[RecommendationType]
    ) -> str:
        """캐시 키를 생성합니다."""
        types_str = "_".join([t.value for t in recommendation_types])
        return f"{context.user_id}_{context.current_topic}_{types_str}_{context.time_preference}"
    
    def _is_cache_valid(self, timestamp: datetime) -> bool:
        """캐시가 유효한지 확인합니다."""
        ttl = self.cache_ttl_hours * 3600  # 초로 변환
        return (datetime.now(timezone.utc) - timestamp).total_seconds() < ttl
    
    async def get_recommendation_explanation(
        self,
        recommendation: Recommendation,
        context: RecommendationContext
    ) -> str:
        """추천에 대한 설명을 생성합니다."""
        try:
            explanations = {
                RecommendationType.SIMILAR_RESEARCH: f"This research is similar to your current topic with {recommendation.similarity_score:.2f} similarity score.",
                RecommendationType.RELATED_TOPICS: f"Based on your research interests, you might find {recommendation.topic} interesting.",
                RecommendationType.CONTINUATION: f"You can build upon your previous research on {recommendation.topic}.",
                RecommendationType.CROSS_DOMAIN: f"Explore how your research applies to {recommendation.metadata.get('cross_domain', 'other domains')}.",
                RecommendationType.TRENDING: f"This is a trending topic with {recommendation.metadata.get('trending_count', 0)} recent studies."
            }
            
            base_explanation = explanations.get(recommendation.type, "This recommendation is based on your research patterns.")
            
            if recommendation.confidence > 0.8:
                confidence_text = "high confidence"
            elif recommendation.confidence > 0.5:
                confidence_text = "medium confidence"
            else:
                confidence_text = "low confidence"
            
            return f"{base_explanation} (Recommended with {confidence_text})"
            
        except Exception as e:
            logger.error(f"Failed to generate recommendation explanation: {e}")
            return "This recommendation is based on your research patterns and preferences."
    
    def get_recommender_stats(self) -> Dict[str, Any]:
        """추천 시스템 통계를 반환합니다."""
        return {
            'cached_recommendations': len(self.recommendation_cache),
            'config': self.config,
            'cache_ttl_hours': self.cache_ttl_hours
        }
