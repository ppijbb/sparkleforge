#!/usr/bin/env python3
"""
User Profiler for Local Researcher Project (v2.0 - 8대 혁신)

Production-grade 사용자 프로파일 및 선호도 학습 시스템.
사용자별 연구 패턴을 학습하고 과거 연구를 기억하여
점점 더 개인화된 결과를 제공합니다.

2025년 10월 최신 기술 스택:
- MCP tools for data collection
- Machine learning for pattern recognition
- Production-grade user modeling
- Privacy-preserving analytics
"""

import asyncio
import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict, Counter
import numpy as np
from pathlib import Path

# MCP integration
import sys
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core.mcp_integration import execute_tool
from src.core.llm_manager import execute_llm_task, TaskType
from src.core.logging_config import get_logger

logger = get_logger(__name__)


class ResearchDepth(Enum):
    """연구 깊이."""
    BASIC = "basic"
    DETAILED = "detailed"
    COMPREHENSIVE = "comprehensive"


class LanguagePreference(Enum):
    """언어 선호도."""
    KOREAN = "korean"
    ENGLISH = "english"
    JAPANESE = "japanese"
    CHINESE = "chinese"


@dataclass
class UserProfile:
    """사용자 프로파일."""
    user_id: str
    preferred_topics: List[str]
    preferred_sources: List[str]
    avg_research_depth: ResearchDepth
    language_preference: LanguagePreference
    citation_style: str
    research_frequency: str  # daily, weekly, monthly, occasional
    preferred_output_format: str  # report, summary, detailed, visual
    domain_expertise: List[str]
    research_patterns: Dict[str, Any]
    learning_preferences: Dict[str, Any]
    created_at: datetime
    last_updated: datetime
    confidence_score: float = 0.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환."""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        data['last_updated'] = self.last_updated.isoformat()
        data['avg_research_depth'] = self.avg_research_depth.value
        data['language_preference'] = self.language_preference.value
        return data


@dataclass
class ResearchSession:
    """연구 세션."""
    session_id: str
    user_id: str
    topic: str
    start_time: datetime
    end_time: Optional[datetime]
    research_depth: ResearchDepth
    sources_used: List[str]
    output_format: str
    satisfaction_score: Optional[float]
    feedback: Optional[str]
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class UserProfiler:
    """
    Production-grade 사용자 프로파일 및 선호도 학습 시스템.
    
    Features:
    - 사용자별 연구 주제 선호도 추적
    - 자주 사용하는 출처 타입 학습
    - 선호하는 보고서 형식 기억
    - 피드백 기반 개인화 조정
    """
    
    def __init__(self, storage_path: str = "./user_profiles"):
        """
        사용자 프로파일러 초기화.
        
        Args:
            storage_path: 프로파일 저장 경로
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # 사용자 프로파일 캐시
        self.profiles: Dict[str, UserProfile] = {}
        
        # 연구 세션 추적
        self.active_sessions: Dict[str, ResearchSession] = {}
        
        # 학습 통계
        self.learning_stats = {
            'total_users': 0,
            'total_sessions': 0,
            'avg_profile_confidence': 0.0,
            'last_learning_update': datetime.now(timezone.utc)
        }
        
        logger.info("UserProfiler initialized")
    
    async def get_or_create_profile(self, user_id: str) -> UserProfile:
        """사용자 프로파일을 가져오거나 생성합니다."""
        try:
            # 캐시에서 확인
            if user_id in self.profiles:
                return self.profiles[user_id]
            
            # 파일에서 로드
            profile = await self._load_profile(user_id)
            
            if not profile:
                # 새 프로파일 생성
                profile = await self._create_new_profile(user_id)
                await self._save_profile(profile)
            
            # 캐시에 저장
            self.profiles[user_id] = profile
            
            return profile
            
        except Exception as e:
            logger.error(f"Failed to get/create profile for user {user_id}: {e}")
            return await self._create_new_profile(user_id)
    
    async def _create_new_profile(self, user_id: str) -> UserProfile:
        """새 사용자 프로파일을 생성합니다."""
        now = datetime.now(timezone.utc)
        
        return UserProfile(
            user_id=user_id,
            preferred_topics=[],
            preferred_sources=[],
            avg_research_depth=ResearchDepth.DETAILED,
            language_preference=LanguagePreference.KOREAN,
            citation_style="apa",
            research_frequency="monthly",
            preferred_output_format="report",
            domain_expertise=[],
            research_patterns={},
            learning_preferences={},
            created_at=now,
            last_updated=now,
            confidence_score=0.0
        )
    
    async def _load_profile(self, user_id: str) -> Optional[UserProfile]:
        """파일에서 프로파일을 로드합니다."""
        try:
            profile_file = self.storage_path / f"{user_id}_profile.json"
            
            if not profile_file.exists():
                return None
            
            data = json.loads(profile_file.read_text(encoding='utf-8'))
            
            return UserProfile(
                user_id=data['user_id'],
                preferred_topics=data['preferred_topics'],
                preferred_sources=data['preferred_sources'],
                avg_research_depth=ResearchDepth(data['avg_research_depth']),
                language_preference=LanguagePreference(data['language_preference']),
                citation_style=data['citation_style'],
                research_frequency=data['research_frequency'],
                preferred_output_format=data['preferred_output_format'],
                domain_expertise=data['domain_expertise'],
                research_patterns=data['research_patterns'],
                learning_preferences=data['learning_preferences'],
                created_at=datetime.fromisoformat(data['created_at']),
                last_updated=datetime.fromisoformat(data['last_updated']),
                confidence_score=data.get('confidence_score', 0.0),
                metadata=data.get('metadata', {})
            )
            
        except Exception as e:
            logger.error(f"Failed to load profile for user {user_id}: {e}")
            return None
    
    async def _save_profile(self, profile: UserProfile) -> bool:
        """프로파일을 파일에 저장합니다."""
        try:
            profile_file = self.storage_path / f"{profile.user_id}_profile.json"
            profile_file.write_text(
                json.dumps(profile.to_dict(), ensure_ascii=False, indent=2),
                encoding='utf-8'
            )
            
            logger.debug(f"Profile saved for user {profile.user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save profile for user {profile.user_id}: {e}")
            return False
    
    async def start_research_session(
        self,
        user_id: str,
        topic: str,
        research_depth: ResearchDepth = ResearchDepth.DETAILED,
        output_format: str = "report"
    ) -> str:
        """연구 세션을 시작합니다."""
        try:
            session_id = f"session_{user_id}_{int(datetime.now().timestamp())}"
            
            session = ResearchSession(
                session_id=session_id,
                user_id=user_id,
                topic=topic,
                start_time=datetime.now(timezone.utc),
                end_time=None,
                research_depth=research_depth,
                sources_used=[],
                output_format=output_format,
                satisfaction_score=None,
                feedback=None
            )
            
            self.active_sessions[session_id] = session
            
            logger.info(f"Research session started: {session_id}")
            return session_id
            
        except Exception as e:
            logger.error(f"Failed to start research session: {e}")
            return ""
    
    async def end_research_session(
        self,
        session_id: str,
        sources_used: List[str],
        satisfaction_score: Optional[float] = None,
        feedback: Optional[str] = None
    ) -> bool:
        """연구 세션을 종료합니다."""
        try:
            if session_id not in self.active_sessions:
                logger.warning(f"Session not found: {session_id}")
                return False
            
            session = self.active_sessions[session_id]
            session.end_time = datetime.now(timezone.utc)
            session.sources_used = sources_used
            session.satisfaction_score = satisfaction_score
            session.feedback = feedback
            
            # 프로파일 업데이트
            await self._update_profile_from_session(session)
            
            # 세션 저장
            await self._save_session(session)
            
            # 활성 세션에서 제거
            del self.active_sessions[session_id]
            
            logger.info(f"Research session ended: {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to end research session {session_id}: {e}")
            return False
    
    async def _update_profile_from_session(self, session: ResearchSession) -> None:
        """세션 정보로 프로파일을 업데이트합니다."""
        try:
            profile = await self.get_or_create_profile(session.user_id)
            
            # 주제 선호도 업데이트
            if session.topic not in profile.preferred_topics:
                profile.preferred_topics.append(session.topic)
            
            # 출처 선호도 업데이트
            for source in session.sources_used:
                if source not in profile.preferred_sources:
                    profile.preferred_sources.append(source)
            
            # 연구 깊이 패턴 업데이트
            depth_key = f"depth_{session.research_depth.value}"
            profile.research_patterns[depth_key] = profile.research_patterns.get(depth_key, 0) + 1
            
            # 출력 형식 선호도 업데이트
            format_key = f"format_{session.output_format}"
            profile.research_patterns[format_key] = profile.research_patterns.get(format_key, 0) + 1
            
            # 만족도 기반 조정
            if session.satisfaction_score is not None:
                await self._adjust_preferences_from_feedback(profile, session)
            
            # 신뢰도 점수 업데이트
            profile.confidence_score = self._calculate_profile_confidence(profile)
            profile.last_updated = datetime.now(timezone.utc)
            
            # 프로파일 저장
            await self._save_profile(profile)
            self.profiles[session.user_id] = profile
            
        except Exception as e:
            logger.error(f"Failed to update profile from session: {e}")
    
    async def _adjust_preferences_from_feedback(self, profile: UserProfile, session: ResearchSession) -> None:
        """피드백을 기반으로 선호도를 조정합니다."""
        try:
            if not session.feedback:
                return
            
            # LLM을 사용한 피드백 분석
            feedback_prompt = f"""
            Analyze the following user feedback and extract preferences:
            
            Topic: {session.topic}
            Research Depth: {session.research_depth.value}
            Output Format: {session.output_format}
            Satisfaction Score: {session.satisfaction_score}
            Feedback: {session.feedback}
            
            Extract:
            1. Preferred research depth (basic, detailed, comprehensive)
            2. Preferred output format (report, summary, detailed, visual)
            3. Domain interests
            4. Source preferences
            5. Language preferences
            
            Return JSON format:
            {{
                "preferred_depth": "detailed",
                "preferred_format": "report",
                "domain_interests": ["technology", "science"],
                "source_preferences": ["academic", "news"],
                "language_preference": "korean"
            }}
            """
            
            result = await execute_llm_task(
                prompt=feedback_prompt,
                task_type=TaskType.ANALYSIS,
                system_message="You are an expert at analyzing user feedback and extracting preferences."
            )
            
            analysis = json.loads(result.content)
            
            # 선호도 업데이트
            if 'preferred_depth' in analysis:
                depth_mapping = {
                    'basic': ResearchDepth.BASIC,
                    'detailed': ResearchDepth.DETAILED,
                    'comprehensive': ResearchDepth.COMPREHENSIVE
                }
                if analysis['preferred_depth'] in depth_mapping:
                    profile.avg_research_depth = depth_mapping[analysis['preferred_depth']]
            
            if 'preferred_format' in analysis:
                profile.preferred_output_format = analysis['preferred_format']
            
            if 'domain_interests' in analysis:
                for domain in analysis['domain_interests']:
                    if domain not in profile.domain_expertise:
                        profile.domain_expertise.append(domain)
            
            if 'source_preferences' in analysis:
                for source in analysis['source_preferences']:
                    if source not in profile.preferred_sources:
                        profile.preferred_sources.append(source)
            
            if 'language_preference' in analysis:
                lang_mapping = {
                    'korean': LanguagePreference.KOREAN,
                    'english': LanguagePreference.ENGLISH,
                    'japanese': LanguagePreference.JAPANESE,
                    'chinese': LanguagePreference.CHINESE
                }
                if analysis['language_preference'] in lang_mapping:
                    profile.language_preference = lang_mapping[analysis['language_preference']]
            
        except Exception as e:
            logger.error(f"Failed to adjust preferences from feedback: {e}")
    
    def _calculate_profile_confidence(self, profile: UserProfile) -> float:
        """프로파일 신뢰도를 계산합니다."""
        try:
            # 데이터 풍부도 기반 신뢰도
            data_richness = 0.0
            
            # 주제 선호도 (최대 0.3)
            if profile.preferred_topics:
                data_richness += min(0.3, len(profile.preferred_topics) * 0.05)
            
            # 출처 선호도 (최대 0.2)
            if profile.preferred_sources:
                data_richness += min(0.2, len(profile.preferred_sources) * 0.05)
            
            # 도메인 전문성 (최대 0.2)
            if profile.domain_expertise:
                data_richness += min(0.2, len(profile.domain_expertise) * 0.05)
            
            # 연구 패턴 (최대 0.3)
            if profile.research_patterns:
                pattern_count = sum(profile.research_patterns.values())
                data_richness += min(0.3, pattern_count * 0.01)
            
            # 시간 기반 신뢰도 (최근 업데이트일수록 높음)
            days_since_update = (datetime.now(timezone.utc) - profile.last_updated).days
            time_factor = max(0.5, 1.0 - (days_since_update / 30))  # 30일 기준
            
            confidence = data_richness * time_factor
            return min(1.0, max(0.0, confidence))
            
        except Exception as e:
            logger.error(f"Failed to calculate profile confidence: {e}")
            return 0.0
    
    async def _save_session(self, session: ResearchSession) -> bool:
        """세션을 저장합니다."""
        try:
            session_file = self.storage_path / f"{session.user_id}_sessions.json"
            
            # 기존 세션 로드
            sessions = []
            if session_file.exists():
                sessions = json.loads(session_file.read_text(encoding='utf-8'))
            
            # 새 세션 추가
            session_data = {
                'session_id': session.session_id,
                'user_id': session.user_id,
                'topic': session.topic,
                'start_time': session.start_time.isoformat(),
                'end_time': session.end_time.isoformat() if session.end_time else None,
                'research_depth': session.research_depth.value,
                'sources_used': session.sources_used,
                'output_format': session.output_format,
                'satisfaction_score': session.satisfaction_score,
                'feedback': session.feedback,
                'metadata': session.metadata
            }
            
            sessions.append(session_data)
            
            # 최근 100개 세션만 유지
            if len(sessions) > 100:
                sessions = sessions[-100:]
            
            # 저장
            session_file.write_text(
                json.dumps(sessions, ensure_ascii=False, indent=2),
                encoding='utf-8'
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to save session: {e}")
            return False
    
    async def get_user_recommendations(self, user_id: str) -> Dict[str, Any]:
        """사용자 맞춤 추천을 생성합니다."""
        try:
            profile = await self.get_or_create_profile(user_id)
            
            recommendations = {
                'suggested_topics': profile.preferred_topics[:5],
                'recommended_sources': profile.preferred_sources[:5],
                'suggested_depth': profile.avg_research_depth.value,
                'recommended_format': profile.preferred_output_format,
                'domain_suggestions': profile.domain_expertise[:3],
                'confidence_score': profile.confidence_score,
                'personalization_level': self._get_personalization_level(profile)
            }
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Failed to get user recommendations: {e}")
            return {}
    
    def _get_personalization_level(self, profile: UserProfile) -> str:
        """개인화 수준을 반환합니다."""
        confidence = profile.confidence_score
        
        if confidence >= 0.8:
            return "high"
        elif confidence >= 0.5:
            return "medium"
        else:
            return "low"
    
    async def get_learning_insights(self) -> Dict[str, Any]:
        """학습 인사이트를 반환합니다."""
        try:
            # 모든 프로파일 로드
            profile_files = list(self.storage_path.glob("*_profile.json"))
            
            insights = {
                'total_users': len(profile_files),
                'avg_confidence': 0.0,
                'popular_topics': [],
                'popular_sources': [],
                'depth_preferences': {},
                'format_preferences': {},
                'language_distribution': {},
                'last_updated': datetime.now(timezone.utc).isoformat()
            }
            
            if not profile_files:
                return insights
            
            # 통계 수집
            all_topics = []
            all_sources = []
            depth_counts = Counter()
            format_counts = Counter()
            language_counts = Counter()
            confidence_scores = []
            
            for profile_file in profile_files:
                try:
                    data = json.loads(profile_file.read_text(encoding='utf-8'))
                    
                    all_topics.extend(data.get('preferred_topics', []))
                    all_sources.extend(data.get('preferred_sources', []))
                    depth_counts[data.get('avg_research_depth', 'detailed')] += 1
                    format_counts[data.get('preferred_output_format', 'report')] += 1
                    language_counts[data.get('language_preference', 'korean')] += 1
                    confidence_scores.append(data.get('confidence_score', 0.0))
                    
                except Exception as e:
                    logger.warning(f"Failed to process profile file {profile_file}: {e}")
                    continue
            
            # 인사이트 생성
            insights['avg_confidence'] = np.mean(confidence_scores) if confidence_scores else 0.0
            insights['popular_topics'] = [topic for topic, count in Counter(all_topics).most_common(10)]
            insights['popular_sources'] = [source for source, count in Counter(all_sources).most_common(10)]
            insights['depth_preferences'] = dict(depth_counts)
            insights['format_preferences'] = dict(format_counts)
            insights['language_distribution'] = dict(language_counts)
            
            return insights
            
        except Exception as e:
            logger.error(f"Failed to get learning insights: {e}")
            return {}
    
    def get_profiler_stats(self) -> Dict[str, Any]:
        """프로파일러 통계를 반환합니다."""
        return {
            'cached_profiles': len(self.profiles),
            'active_sessions': len(self.active_sessions),
            'storage_path': str(self.storage_path),
            'learning_stats': self.learning_stats
        }
