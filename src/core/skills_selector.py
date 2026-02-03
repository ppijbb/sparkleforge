"""
Skills Selector - Automatic Skills identification and selection

작업 요청 시 관련 Skills를 자동으로 감지 및 로드하는 시스템.
Semantic matching과 의존성 그래프를 활용.
"""

import logging
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass
import re

from src.core.skills_manager import get_skill_manager, SkillManager
from src.core.skills_loader import Skill, SkillMetadata

logger = logging.getLogger(__name__)


@dataclass
class SkillMatch:
    """Skill 매칭 결과."""
    skill_id: str
    score: float
    reasons: List[str]
    metadata: SkillMetadata


class SkillSelector:
    """자동 Skills 식별 및 선택 클래스."""
    
    def __init__(self, skill_manager: Optional[SkillManager] = None):
        """초기화."""
        self.skill_manager = skill_manager or get_skill_manager()
        
        # 키워드 기반 매칭을 위한 키워드 맵
        self.keyword_map = self._build_keyword_map()
    
    def select_skills_for_task(self, query: str, max_skills: int = 5) -> List[SkillMatch]:
        """작업 쿼리에 필요한 Skills 자동 선택."""
        logger.info(f"Selecting skills for task: {query[:100]}")
        
        # 1. 키워드 기반 매칭
        keyword_matches = self._match_by_keywords(query)
        
        # 2. 태그 기반 매칭
        tag_matches = self._match_by_tags(query)
        
        # 3. 설명 기반 매칭 (semantic matching 간단 버전)
        description_matches = self._match_by_description(query)
        
        # 4. 점수 통합
        all_matches = self._merge_matches(keyword_matches, tag_matches, description_matches)
        
        # 5. 상위 N개 선택
        top_matches = sorted(all_matches, key=lambda x: x.score, reverse=True)[:max_skills]
        
        # 6. 의존성 추가
        selected_skills = self._add_dependencies(top_matches)
        
        logger.info(f"Selected {len(selected_skills)} skills: {[s.skill_id for s in selected_skills]}")
        return selected_skills
    
    def _build_keyword_map(self) -> Dict[str, List[str]]:
        """키워드 맵 구축."""
        return {
            "research_planner": [
                "plan", "planning", "strategy", "objective", "goal", "research plan",
                "계획", "전략", "목표", "연구 계획"
            ],
            "research_executor": [
                "search", "find", "gather", "execute", "research", "investigate",
                "검색", "수집", "실행", "연구", "조사"
            ],
            "evaluator": [
                "verify", "validate", "check", "evaluate", "assess", "quality",
                "검증", "평가", "확인", "품질"
            ],
            "synthesizer": [
                "synthesize", "summarize", "report", "generate", "create", "final",
                "종합", "요약", "리포트", "생성", "최종"
            ]
        }
    
    def _match_by_keywords(self, query: str) -> List[SkillMatch]:
        """키워드 기반 매칭."""
        query_lower = query.lower()
        matches = []
        
        for skill_id, keywords in self.keyword_map.items():
            skill_metadata = self.skill_manager.get_skill_by_id(skill_id)
            if not skill_metadata or not skill_metadata.enabled:
                continue
            
            # 키워드 매칭 점수 계산
            matched_keywords = []
            score = 0.0
            
            for keyword in keywords:
                if keyword.lower() in query_lower:
                    matched_keywords.append(keyword)
                    score += 1.0
            
            if score > 0:
                # 정규화 (최대 3점)
                normalized_score = min(score / 3.0, 1.0)
                matches.append(SkillMatch(
                    skill_id=skill_id,
                    score=normalized_score * 0.4,  # 40% 가중치
                    reasons=[f"Keywords matched: {', '.join(matched_keywords[:3])}"],
                    metadata=skill_metadata
                ))
        
        return matches
    
    def _match_by_tags(self, query: str) -> List[SkillMatch]:
        """태그 기반 매칭."""
        query_lower = query.lower()
        matches = []
        
        all_skills = self.skill_manager.get_all_skills(enabled_only=True)
        
        for skill_metadata in all_skills:
            score = 0.0
            matched_tags = []
            
            for tag in skill_metadata.tags:
                if tag.lower() in query_lower:
                    matched_tags.append(tag)
                    score += 1.0
            
            if score > 0:
                normalized_score = min(score / len(skill_metadata.tags) if skill_metadata.tags else 1.0, 1.0)
                matches.append(SkillMatch(
                    skill_id=skill_metadata.skill_id,
                    score=normalized_score * 0.3,  # 30% 가중치
                    reasons=[f"Tags matched: {', '.join(matched_tags)}"],
                    metadata=skill_metadata
                ))
        
        return matches
    
    def _match_by_description(self, query: str) -> List[SkillMatch]:
        """설명 기반 매칭 (간단한 semantic matching)."""
        query_words = set(re.findall(r'\b\w+\b', query.lower()))
        matches = []
        
        all_skills = self.skill_manager.get_all_skills(enabled_only=True)
        
        for skill_metadata in all_skills:
            # 설명과 태그에서 단어 추출
            description_words = set(re.findall(r'\b\w+\b', skill_metadata.description.lower()))
            capability_words = set()
            for cap in skill_metadata.capabilities:
                capability_words.update(re.findall(r'\b\w+\b', cap.lower()))
            
            # 공통 단어 찾기
            common_words = query_words.intersection(description_words.union(capability_words))
            
            if common_words:
                # 점수: 공통 단어 비율
                score = len(common_words) / max(len(query_words), 1)
                matches.append(SkillMatch(
                    skill_id=skill_metadata.skill_id,
                    score=score * 0.3,  # 30% 가중치
                    reasons=[f"Description matched: {len(common_words)} common words"],
                    metadata=skill_metadata
                ))
        
        return matches
    
    def _merge_matches(self, *match_lists: List[SkillMatch]) -> List[SkillMatch]:
        """여러 매칭 결과 통합."""
        skill_scores: Dict[str, Tuple[float, List[str], SkillMetadata]] = {}
        
        for match_list in match_lists:
            for match in match_list:
                if match.skill_id in skill_scores:
                    # 점수 누적, 이유 추가
                    old_score, old_reasons, metadata = skill_scores[match.skill_id]
                    skill_scores[match.skill_id] = (
                        old_score + match.score,
                        old_reasons + match.reasons,
                        metadata
                    )
                else:
                    skill_scores[match.skill_id] = (
                        match.score,
                        match.reasons,
                        match.metadata
                    )
        
        # SkillMatch 객체로 변환
        merged = []
        for skill_id, (total_score, reasons, metadata) in skill_scores.items():
            merged.append(SkillMatch(
                skill_id=skill_id,
                score=min(total_score, 1.0),  # 최대 1.0
                reasons=reasons[:5],  # 최대 5개 이유만
                metadata=metadata
            ))
        
        return merged
    
    def _add_dependencies(self, matches: List[SkillMatch]) -> List[SkillMatch]:
        """의존성 그래프를 따라 필수 Skills 추가."""
        selected_ids = {m.skill_id for m in matches}
        all_selected = list(matches)
        
        # 의존성 그래프 가져오기
        dependency_graph = self.skill_manager.get_skill_dependency_graph()
        
        # BFS로 의존성 추가
        to_check = list(selected_ids)
        while to_check:
            current_id = to_check.pop(0)
            
            # 의존성 확인
            dependencies = dependency_graph.get(current_id, [])
            for dep_id in dependencies:
                if dep_id not in selected_ids:
                    # 의존성 Skill 로드
                    dep_metadata = self.skill_manager.get_skill_by_id(dep_id)
                    if dep_metadata and dep_metadata.enabled:
                        all_selected.append(SkillMatch(
                            skill_id=dep_id,
                            score=0.5,  # 의존성이므로 중간 점수
                            reasons=[f"Required dependency of {current_id}"],
                            metadata=dep_metadata
                        ))
                        selected_ids.add(dep_id)
                        to_check.append(dep_id)
        
        return all_selected
    
    def get_skill_stack_for_task(self, query: str) -> List[Skill]:
        """작업에 필요한 Skills 스택 반환 (Skill 객체로)."""
        matches = self.select_skills_for_task(query)
        
        # Skill 객체 로드
        skills = []
        for match in matches:
            skill = self.skill_manager.load_skill(match.skill_id)
            if skill:
                skills.append(skill)
        
        # 의존성도 함께 로드
        skill_ids = {s.metadata.skill_id for s in skills}
        dependency_graph = self.skill_manager.get_skill_dependency_graph()
        
        to_load = list(skill_ids)
        while to_load:
            current_id = to_load.pop(0)
            dependencies = dependency_graph.get(current_id, [])
            for dep_id in dependencies:
                if dep_id not in skill_ids:
                    dep_skill = self.skill_manager.load_skill(dep_id)
                    if dep_skill:
                        skills.append(dep_skill)
                        skill_ids.add(dep_id)
                        to_load.append(dep_id)
        
        return skills


# 전역 SkillSelector 인스턴스
_skill_selector: Optional[SkillSelector] = None


def get_skill_selector() -> SkillSelector:
    """전역 SkillSelector 인스턴스 반환."""
    global _skill_selector
    if _skill_selector is None:
        _skill_selector = SkillSelector()
    return _skill_selector

