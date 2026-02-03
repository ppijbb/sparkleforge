"""
Skills Composer - Skills composability engine

여러 Skills를 조합하여 사용하는 메커니즘 구현.
Skills 간 통신 인터페이스 정의 및 실행 순서 최적화.
"""

import logging
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque

from src.core.skills_manager import get_skill_manager, SkillManager
from src.core.skills_loader import Skill

logger = logging.getLogger(__name__)


class SkillRelation(Enum):
    """Skill 간 관계."""
    DEPENDS_ON = "depends_on"  # 의존성
    PROVIDES = "provides"  # 제공
    CONFLICTS = "conflicts"  # 충돌
    ENHANCES = "enhances"  # 향상


@dataclass
class SkillStack:
    """Skills 스택."""
    skills: List[Skill]
    execution_order: List[str]  # skill_id 순서
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_skill(self, skill: Skill, position: Optional[int] = None):
        """Skill 추가."""
        if skill.metadata.skill_id in self.execution_order:
            logger.warning(f"Skill {skill.metadata.skill_id} already in stack")
            return
        
        if position is None:
            self.skills.append(skill)
            self.execution_order.append(skill.metadata.skill_id)
        else:
            self.skills.insert(position, skill)
            self.execution_order.insert(position, skill.metadata.skill_id)
    
    def remove_skill(self, skill_id: str):
        """Skill 제거."""
        if skill_id in self.execution_order:
            idx = self.execution_order.index(skill_id)
            self.execution_order.remove(skill_id)
            self.skills.pop(idx)


class SkillsComposer:
    """Skills 컴포저빌리티 엔진."""
    
    def __init__(self, skill_manager: Optional[SkillManager] = None):
        """초기화."""
        self.skill_manager = skill_manager or get_skill_manager()
        self.communication_bus: Dict[str, Any] = {}  # Skills 간 통신 버스
    
    def compose_skill_stack(self, skill_ids: List[str]) -> Optional[SkillStack]:
        """Skills 스택 구성."""
        # 1. 의존성 확인 및 추가
        all_skills = self._resolve_dependencies(skill_ids)
        
        # 2. 실행 순서 최적화
        execution_order = self._optimize_execution_order(all_skills)
        
        # 3. Skills 로드
        loaded_skills = []
        for skill_id in execution_order:
            skill = self.skill_manager.load_skill(skill_id)
            if skill:
                loaded_skills.append(skill)
            else:
                logger.warning(f"Failed to load skill: {skill_id}")
        
        if not loaded_skills:
            return None
        
        # 4. SkillStack 생성
        stack = SkillStack(
            skills=loaded_skills,
            execution_order=execution_order,
            metadata={
                "original_skills": skill_ids,
                "total_skills": len(loaded_skills),
                "composed_at": str(logging.Formatter().formatTime(logging.LogRecord(
                    name="", level=0, pathname="", lineno=0, msg="", args=(), exc_info=None
                )))
            }
        )
        
        logger.info(f"✅ Composed skill stack with {len(loaded_skills)} skills")
        return stack
    
    def _resolve_dependencies(self, skill_ids: List[str]) -> Set[str]:
        """의존성 해결."""
        resolved = set()
        to_resolve = deque(skill_ids)
        
        while to_resolve:
            skill_id = to_resolve.popleft()
            if skill_id in resolved:
                continue
            
            resolved.add(skill_id)
            
            # 의존성 확인
            dependency_graph = self.skill_manager.get_skill_dependency_graph()
            dependencies = dependency_graph.get(skill_id, [])
            
            for dep_id in dependencies:
                if dep_id not in resolved:
                    to_resolve.append(dep_id)
        
        return resolved
    
    def _optimize_execution_order(self, skill_ids: Set[str]) -> List[str]:
        """실행 순서 최적화 (위상 정렬)."""
        dependency_graph = self.skill_manager.get_skill_dependency_graph()
        
        # 인접 리스트 구축
        graph: Dict[str, List[str]] = {sid: [] for sid in skill_ids}
        in_degree: Dict[str, int] = {sid: 0 for sid in skill_ids}
        
        for skill_id in skill_ids:
            dependencies = dependency_graph.get(skill_id, [])
            for dep_id in dependencies:
                if dep_id in skill_ids:
                    graph[dep_id].append(skill_id)
                    in_degree[skill_id] += 1
        
        # 위상 정렬 (Kahn's algorithm)
        queue = deque([sid for sid in skill_ids if in_degree[sid] == 0])
        result = []
        
        while queue:
            current = queue.popleft()
            result.append(current)
            
            for neighbor in graph[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        # 순환 의존성 체크
        if len(result) != len(skill_ids):
            logger.warning("Circular dependency detected, using original order")
            return list(skill_ids)
        
        return result
    
    def communicate(self, from_skill_id: str, to_skill_id: str, message: Dict[str, Any]):
        """Skills 간 통신."""
        channel = f"{from_skill_id}->{to_skill_id}"
        self.communication_bus[channel] = message
        logger.debug(f"Message from {from_skill_id} to {to_skill_id}: {message.get('type', 'unknown')}")
    
    def receive_message(self, skill_id: str) -> List[Dict[str, Any]]:
        """특정 Skill이 받을 메시지 조회."""
        messages = []
        for channel, message in self.communication_bus.items():
            if channel.endswith(f"->{skill_id}"):
                messages.append(message)
        return messages
    
    def clear_communication_bus(self):
        """통신 버스 초기화."""
        self.communication_bus.clear()
    
    def validate_stack(self, stack: SkillStack) -> Tuple[bool, List[str]]:
        """Skills 스택 검증."""
        issues = []
        
        # 1. 의존성 검증
        for i, skill in enumerate(stack.skills):
            dependencies = skill.metadata.dependencies
            for dep_id in dependencies:
                # 의존성이 스택에 있고, 현재 skill보다 먼저 실행되어야 함
                if dep_id in stack.execution_order:
                    dep_idx = stack.execution_order.index(dep_id)
                    if dep_idx >= i:
                        issues.append(f"Dependency order issue: {skill.metadata.skill_id} requires {dep_id}")
        
        # 2. 충돌 검증 (현재는 간단히 체크)
        skill_categories = {}
        for skill in stack.skills:
            category = skill.metadata.category
            if category in skill_categories and skill_categories[category] != skill.metadata.skill_id:
                # 같은 카테고리에 여러 skill이 있으면 경고 (충돌 가능성)
                logger.debug(f"Multiple skills in category {category}: potential conflict")
        
        is_valid = len(issues) == 0
        return is_valid, issues
    
    def optimize_stack(self, stack: SkillStack) -> SkillStack:
        """스택 최적화 (중복 제거, 순서 재조정)."""
        # 실행 순서 재최적화
        optimized_order = self._optimize_execution_order(set(stack.execution_order))
        
        # Skills 재정렬
        skill_map = {s.metadata.skill_id: s for s in stack.skills}
        optimized_skills = [skill_map[sid] for sid in optimized_order if sid in skill_map]
        
        return SkillStack(
            skills=optimized_skills,
            execution_order=optimized_order,
            metadata=stack.metadata
        )


# 전역 SkillsComposer 인스턴스
_skills_composer: Optional[SkillsComposer] = None


def get_skills_composer() -> SkillsComposer:
    """전역 SkillsComposer 인스턴스 반환."""
    global _skills_composer
    if _skills_composer is None:
        _skills_composer = SkillsComposer()
    return _skills_composer

