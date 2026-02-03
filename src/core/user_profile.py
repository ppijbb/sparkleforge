"""
User Profile Management

사용자별 선호도 학습, 과거 세션 패턴 분석, 컨텍스트 우선순위 개인화.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import json
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class UserPreferences:
    """사용자 선호도."""
    preferred_languages: List[str] = field(default_factory=list)
    preferred_domains: List[str] = field(default_factory=list)
    context_priority_weights: Dict[str, float] = field(default_factory=dict)
    output_format_preference: str = "report"
    detail_level: str = "detailed"  # minimal, standard, detailed


@dataclass
class DomainExpertise:
    """도메인별 전문성."""
    domain: str
    expertise_level: float  # 0.0 ~ 1.0
    topics_covered: List[str] = field(default_factory=list)
    last_accessed: str = ""
    session_count: int = 0


@dataclass
class SessionPattern:
    """세션 패턴."""
    query_patterns: List[str] = field(default_factory=list)
    common_tasks: List[str] = field(default_factory=list)
    average_session_duration: float = 0.0
    preferred_agents: List[str] = field(default_factory=list)
    success_rate: float = 0.0


@dataclass
class UserProfile:
    """사용자 프로필."""
    user_id: str
    created_at: str
    last_updated: str
    preferences: UserPreferences
    domain_expertise: List[DomainExpertise]
    session_patterns: SessionPattern
    total_sessions: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환."""
        return {
            "user_id": self.user_id,
            "created_at": self.created_at,
            "last_updated": self.last_updated,
            "preferences": asdict(self.preferences),
            "domain_expertise": [asdict(de) for de in self.domain_expertise],
            "session_patterns": asdict(self.session_patterns),
            "total_sessions": self.total_sessions,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserProfile":
        """딕셔너리에서 생성."""
        return cls(
            user_id=data["user_id"],
            created_at=data["created_at"],
            last_updated=data["last_updated"],
            preferences=UserPreferences(**data["preferences"]),
            domain_expertise=[DomainExpertise(**de) for de in data["domain_expertise"]],
            session_patterns=SessionPattern(**data["session_patterns"]),
            total_sessions=data.get("total_sessions", 0),
            metadata=data.get("metadata", {})
        )


class UserProfileManager:
    """
    사용자 프로필 관리자.
    
    사용자별 선호도 학습, 세션 패턴 분석, 컨텍스트 우선순위 개인화를 담당합니다.
    """
    
    def __init__(self, storage_path: str = "./storage/user_profiles"):
        """
        초기화.
        
        Args:
            storage_path: 프로필 저장 경로
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.profiles: Dict[str, UserProfile] = {}
        
        logger.info(f"UserProfileManager initialized at {self.storage_path}")
    
    def get_profile(self, user_id: str) -> UserProfile:
        """
        사용자 프로필 조회 (없으면 생성).
        
        Args:
            user_id: 사용자 ID
            
        Returns:
            UserProfile
        """
        if user_id not in self.profiles:
            # 저장소에서 로드 시도
            profile = self._load_profile(user_id)
            if profile:
                self.profiles[user_id] = profile
            else:
                # 새 프로필 생성
                self.profiles[user_id] = self._create_default_profile(user_id)
        
        return self.profiles[user_id]
    
    def update_profile(
        self,
        user_id: str,
        session_data: Dict[str, Any],
        query: str,
        domain: Optional[str] = None
    ) -> bool:
        """
        세션 데이터를 기반으로 프로필 업데이트.
        
        Args:
            user_id: 사용자 ID
            session_data: 세션 데이터 (AgentState 등)
            query: 사용자 쿼리
            domain: 도메인 (자동 추출 가능)
            
        Returns:
            성공 여부
        """
        try:
            profile = self.get_profile(user_id)
            
            # 세션 수 증가
            profile.total_sessions += 1
            profile.last_updated = datetime.now().isoformat()
            
            # 쿼리 패턴 분석
            self._update_query_patterns(profile, query)
            
            # 도메인 전문성 업데이트
            if domain:
                self._update_domain_expertise(profile, domain, query)
            
            # 선호도 학습
            self._learn_preferences(profile, session_data)
            
            # 세션 패턴 업데이트
            self._update_session_patterns(profile, session_data)
            
            # 저장
            self._save_profile(profile)
            
            logger.debug(f"Profile updated for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update profile for {user_id}: {e}")
            return False
    
    def get_personalized_context_weights(self, user_id: str) -> Dict[str, float]:
        """
        개인화된 컨텍스트 우선순위 가중치 반환.
        
        Args:
            user_id: 사용자 ID
            
        Returns:
            컨텍스트 타입별 가중치
        """
        profile = self.get_profile(user_id)
        
        # 기본 가중치
        default_weights = {
            "system_prompt": 1.0,
            "user_query": 1.0,
            "tool_results": 0.8,
            "agent_memory": 0.6,
            "conversation_history": 0.7,
            "metadata": 0.3
        }
        
        # 사용자 선호도 기반 조정
        user_weights = profile.preferences.context_priority_weights
        if user_weights:
            default_weights.update(user_weights)
        
        return default_weights
    
    def get_domain_expertise(self, user_id: str, domain: str) -> float:
        """
        도메인별 전문성 수준 반환.
        
        Args:
            user_id: 사용자 ID
            domain: 도메인
            
        Returns:
            전문성 수준 (0.0 ~ 1.0)
        """
        profile = self.get_profile(user_id)
        
        for de in profile.domain_expertise:
            if de.domain == domain:
                return de.expertise_level
        
        return 0.0
    
    def _create_default_profile(self, user_id: str) -> UserProfile:
        """기본 프로필 생성."""
        return UserProfile(
            user_id=user_id,
            created_at=datetime.now().isoformat(),
            last_updated=datetime.now().isoformat(),
            preferences=UserPreferences(),
            domain_expertise=[],
            session_patterns=SessionPattern(),
            total_sessions=0
        )
    
    def _load_profile(self, user_id: str) -> Optional[UserProfile]:
        """저장소에서 프로필 로드."""
        try:
            profile_file = self.storage_path / f"{user_id}_profile.json"
            if profile_file.exists():
                data = json.loads(profile_file.read_text(encoding='utf-8'))
                return UserProfile.from_dict(data)
        except Exception as e:
            logger.warning(f"Failed to load profile for {user_id}: {e}")
        return None
    
    def _save_profile(self, profile: UserProfile):
        """프로필 저장."""
        try:
            profile_file = self.storage_path / f"{profile.user_id}_profile.json"
            profile_file.write_text(
                json.dumps(profile.to_dict(), ensure_ascii=False, indent=2),
                encoding='utf-8'
            )
        except Exception as e:
            logger.error(f"Failed to save profile for {profile.user_id}: {e}")
    
    def _update_query_patterns(self, profile: UserProfile, query: str):
        """쿼리 패턴 업데이트."""
        # 간단한 키워드 추출
        keywords = query.lower().split()[:5]  # 상위 5개 단어
        profile.session_patterns.query_patterns.extend(keywords)
        
        # 최근 100개만 유지
        if len(profile.session_patterns.query_patterns) > 100:
            profile.session_patterns.query_patterns = profile.session_patterns.query_patterns[-100:]
    
    def _update_domain_expertise(self, profile: UserProfile, domain: str, query: str):
        """도메인 전문성 업데이트."""
        # 기존 도메인 찾기
        found = False
        for de in profile.domain_expertise:
            if de.domain == domain:
                de.expertise_level = min(1.0, de.expertise_level + 0.05)  # 점진적 증가
                de.session_count += 1
                de.last_accessed = datetime.now().isoformat()
                if query not in de.topics_covered:
                    de.topics_covered.append(query)
                found = True
                break
        
        # 새 도메인 추가
        if not found:
            profile.domain_expertise.append(DomainExpertise(
                domain=domain,
                expertise_level=0.1,
                topics_covered=[query],
                last_accessed=datetime.now().isoformat(),
                session_count=1
            ))
    
    def _learn_preferences(self, profile: UserProfile, session_data: Dict[str, Any]):
        """선호도 학습."""
        # 출력 형식 선호도
        if "final_report" in session_data and session_data["final_report"]:
            # 보고서 형식 분석 (간단한 휴리스틱)
            report = session_data["final_report"]
            if "##" in report or "#" in report:
                profile.preferences.output_format_preference = "markdown"
            elif "```" in report:
                profile.preferences.output_format_preference = "code"
            else:
                profile.preferences.output_format_preference = "report"
        
        # 컨텍스트 우선순위 가중치 학습 (성공적인 세션 기반)
        if not session_data.get("research_failed") and not session_data.get("verification_failed"):
            # 성공적인 세션: 현재 가중치 유지 또는 약간 증가
            # 실패한 세션: 가중치 감소 (향후 구현)
            pass
    
    def _update_session_patterns(self, profile: UserProfile, session_data: Dict[str, Any]):
        """세션 패턴 업데이트."""
        # 사용된 에이전트 추적
        current_agent = session_data.get("current_agent")
        if current_agent and current_agent not in profile.session_patterns.preferred_agents:
            profile.session_patterns.preferred_agents.append(current_agent)
        
        # 성공률 계산
        if profile.total_sessions > 0:
            success_count = profile.total_sessions - (
                1 if session_data.get("research_failed") or session_data.get("verification_failed") else 0
            )
            profile.session_patterns.success_rate = success_count / profile.total_sessions


# 전역 인스턴스
_user_profile_manager: Optional[UserProfileManager] = None


def get_user_profile_manager() -> UserProfileManager:
    """전역 사용자 프로필 관리자 인스턴스 반환."""
    global _user_profile_manager
    if _user_profile_manager is None:
        import os
        storage_path = os.getenv("USER_PROFILE_STORAGE_PATH", "./storage/user_profiles")
        _user_profile_manager = UserProfileManager(storage_path=storage_path)
    return _user_profile_manager

