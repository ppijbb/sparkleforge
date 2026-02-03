"""
Skills Manager - Manage and load Skills

Skills 스캔, 메타데이터 파싱, 필요 시에만 로드 (lazy loading), 캐싱을 담당하는 모듈.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Set
from datetime import datetime
from dataclasses import dataclass, field

from src.core.skills_loader import SkillLoader, Skill, SkillMetadata

logger = logging.getLogger(__name__)


@dataclass
class SkillRegistry:
    """Skills 메타데이터 저장소."""
    version: str
    registry_updated_at: str
    skills: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    skill_categories: Dict[str, List[str]] = field(default_factory=dict)
    skill_dependencies: Dict[str, List[str]] = field(default_factory=dict)


class SkillManager:
    """Skills 전체 관리 클래스."""
    
    def __init__(self, project_root: Optional[Path] = None):
        """초기화."""
        if project_root is None:
            project_root = Path(__file__).parent.parent.parent
        
        self.project_root = Path(project_root)
        self.skills_dir = self.project_root / "skills"
        self.registry_path = self.project_root / "skills_registry.json"
        
        self.loader = SkillLoader(self.project_root)
        self.registry: Optional[SkillRegistry] = None
        self.loaded_skills: Dict[str, Skill] = {}  # 캐시
        self.skill_metadata_cache: Dict[str, SkillMetadata] = {}
        
        # 레지스트리 로드
        self._load_registry()
        # Skills 스캔
        self._scan_skills()
    
    def _load_registry(self):
        """skills_registry.json 로드."""
        if not self.registry_path.exists():
            logger.warning(f"skills_registry.json not found at {self.registry_path}")
            self.registry = SkillRegistry(
                version="1.0.0",
                registry_updated_at=datetime.now().isoformat()
            )
            return
        
        try:
            with open(self.registry_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.registry = SkillRegistry(
                    version=data.get("version", "1.0.0"),
                    registry_updated_at=data.get("registry_updated_at", datetime.now().isoformat()),
                    skills=data.get("skills", {}),
                    skill_categories=data.get("skill_categories", {}),
                    skill_dependencies=data.get("skill_dependencies", {})
                )
            logger.info(f"✅ Loaded skills registry with {len(self.registry.skills)} skills")
        except Exception as e:
            logger.error(f"Failed to load registry: {e}")
            self.registry = SkillRegistry(
                version="1.0.0",
                registry_updated_at=datetime.now().isoformat()
            )
    
    def _scan_skills(self):
        """skills/ 디렉토리 스캔 및 메타데이터 캐싱."""
        if not self.skills_dir.exists():
            logger.warning(f"Skills directory not found: {self.skills_dir}")
            return
        
        scanned_count = 0
        for skill_dir in self.skills_dir.iterdir():
            if not skill_dir.is_dir():
                continue
            
            skill_id = skill_dir.name
            skill_md_path = skill_dir / "SKILL.md"
            
            if not skill_md_path.exists():
                logger.warning(f"SKILL.md not found for skill: {skill_id}")
                continue
            
            try:
                # 레지스트리에 없는 경우 메타데이터만 로드 (빠른 스캔)
                if skill_id in self.registry.skills:
                    # 레지스트리에서 메타데이터 사용
                    reg_data = self.registry.skills[skill_id]
                    metadata = SkillMetadata(
                        skill_id=reg_data.get("skill_id", skill_id),
                        name=reg_data.get("name", skill_id),
                        description=reg_data.get("description", ""),
                        version=reg_data.get("version", "1.0.0"),
                        category=reg_data.get("category", "general"),
                        tags=reg_data.get("tags", []),
                        author=reg_data.get("author", "Unknown"),
                        created_at=reg_data.get("created_at", datetime.now().isoformat()),
                        updated_at=reg_data.get("updated_at", datetime.now().isoformat()),
                        path=reg_data.get("path", f"skills/{skill_id}"),
                        enabled=reg_data.get("enabled", True),
                        dependencies=reg_data.get("dependencies", []),
                        required_tools=reg_data.get("required_tools", []),
                        capabilities=reg_data.get("metadata", {}).get("capabilities", [])
                    )
                else:
                    # SKILL.md에서 메타데이터 빠르게 추출 (전체 로드 아님)
                    skill_content = skill_md_path.read_text(encoding='utf-8')
                    parsed = self.loader._parse_skill_md(skill_content)
                    metadata = self.loader._extract_metadata(skill_id, skill_dir, parsed)
                
                self.skill_metadata_cache[skill_id] = metadata
                scanned_count += 1
                
            except Exception as e:
                logger.error(f"Failed to scan skill {skill_id}: {e}")
        
        logger.info(f"✅ Scanned {scanned_count} skills")
    
    def get_all_skills(self, enabled_only: bool = False) -> List[SkillMetadata]:
        """모든 Skills 메타데이터 반환."""
        skills = list(self.skill_metadata_cache.values())
        if enabled_only:
            skills = [s for s in skills if s.enabled]
        return skills
    
    def get_skill_by_id(self, skill_id: str) -> Optional[SkillMetadata]:
        """Skill ID로 메타데이터 반환."""
        return self.skill_metadata_cache.get(skill_id)
    
    def get_skills_by_category(self, category: str) -> List[SkillMetadata]:
        """카테고리로 Skills 필터링."""
        return [s for s in self.skill_metadata_cache.values() 
                if s.category == category and s.enabled]
    
    def get_skills_by_tag(self, tag: str) -> List[SkillMetadata]:
        """태그로 Skills 필터링."""
        return [s for s in self.skill_metadata_cache.values() 
                if tag in s.tags and s.enabled]
    
    def load_skill(self, skill_id: str, force_reload: bool = False) -> Optional[Skill]:
        """Skill을 로드 (lazy loading with caching)."""
        # 캐시 확인
        if not force_reload and skill_id in self.loaded_skills:
            logger.debug(f"Using cached skill: {skill_id}")
            return self.loaded_skills[skill_id]
        
        # 메타데이터 확인
        if skill_id not in self.skill_metadata_cache:
            logger.error(f"Skill not found: {skill_id}")
            return None
        
        metadata = self.skill_metadata_cache[skill_id]
        if not metadata.enabled:
            logger.warning(f"Skill {skill_id} is disabled")
            return None
        
        # Skill 로드
        skill = self.loader.load_skill(skill_id)
        if skill:
            self.loaded_skills[skill_id] = skill
            logger.info(f"✅ Loaded skill: {skill_id}")
        
        return skill
    
    def load_skill_dependencies(self, skill_id: str) -> List[Skill]:
        """Skill과 그 의존성을 모두 로드."""
        loaded = []
        to_load = [skill_id]
        loaded_ids = set()
        
        while to_load:
            current_id = to_load.pop(0)
            if current_id in loaded_ids:
                continue
            
            loaded_ids.add(current_id)
            
            # Skill 로드
            skill = self.load_skill(current_id)
            if skill:
                loaded.append(skill)
                
                # 의존성 추가
                for dep in skill.metadata.dependencies:
                    if dep not in loaded_ids:
                        to_load.append(dep)
        
        return loaded
    
    def get_skill_instruction(self, skill_id: str) -> Optional[str]:
        """Skill의 instruction만 빠르게 반환 (전체 로드 없이)."""
        skill = self.load_skill(skill_id)
        if skill:
            return skill.instructions
        return None
    
    def get_available_skills(self) -> List[str]:
        """사용 가능한 Skill ID 목록 반환."""
        return [skill_id for skill_id, metadata in self.skill_metadata_cache.items() 
                if metadata.enabled]
    
    def get_skill_dependency_graph(self) -> Dict[str, List[str]]:
        """Skills 의존성 그래프 반환."""
        if self.registry:
            return self.registry.skill_dependencies.copy()
        return {}
    
    def enable_skill(self, skill_id: str) -> bool:
        """Skill 활성화."""
        if skill_id not in self.skill_metadata_cache:
            return False
        
        self.skill_metadata_cache[skill_id].enabled = True
        # 레지스트리도 업데이트
        if self.registry and skill_id in self.registry.skills:
            self.registry.skills[skill_id]["enabled"] = True
        return True
    
    def disable_skill(self, skill_id: str) -> bool:
        """Skill 비활성화."""
        if skill_id not in self.skill_metadata_cache:
            return False
        
        self.skill_metadata_cache[skill_id].enabled = False
        # 레지스트리도 업데이트
        if self.registry and skill_id in self.registry.skills:
            self.registry.skills[skill_id]["enabled"] = False
        return True
    
    def refresh_registry(self):
        """레지스트리 새로고침 (스캔 재수행)."""
        self._scan_skills()
    
    def clear_cache(self):
        """로드된 Skills 캐시 지우기."""
        self.loaded_skills.clear()
        logger.info("✅ Cleared skills cache")


# 전역 SkillManager 인스턴스 (lazy initialization)
_skill_manager: Optional[SkillManager] = None


def get_skill_manager() -> SkillManager:
    """전역 SkillManager 인스턴스 반환."""
    global _skill_manager
    if _skill_manager is None:
        _skill_manager = SkillManager()
    return _skill_manager

