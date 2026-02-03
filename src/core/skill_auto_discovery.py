"""
Skill Auto-Discovery and Hot-Reload System

ERA 영감을 받은 자동 스킬 발견 및 핫 리로드 시스템.
SKILL.md 파일을 자동으로 발견하고 런타임에 동적으로 로드/리로드.

핵심 특징:
- Automatic SKILL.md discovery from filesystem
- Hot-reload on file changes
- Skill context injection for execution
- Skill dependency resolution
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable, Set
from datetime import datetime
from pathlib import Path
from enum import Enum
import hashlib
import threading

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class SkillStatus(Enum):
    """스킬 상태."""
    DISCOVERED = "discovered"
    LOADING = "loading"
    LOADED = "loaded"
    FAILED = "failed"
    DISABLED = "disabled"
    RELOADING = "reloading"


class SkillChangeType(Enum):
    """스킬 변경 유형."""
    ADDED = "added"
    MODIFIED = "modified"
    REMOVED = "removed"


class DiscoveredSkill(BaseModel):
    """발견된 스킬 정보."""
    skill_id: str = Field(description="스킬 고유 ID")
    name: str = Field(default="", description="스킬 이름")
    description: str = Field(default="", description="스킬 설명")
    path: str = Field(description="SKILL.md 파일 경로")
    directory: str = Field(description="스킬 디렉토리")
    
    # 상태
    status: SkillStatus = Field(default=SkillStatus.DISCOVERED)
    last_modified: Optional[datetime] = Field(default=None)
    content_hash: str = Field(default="", description="파일 해시 (변경 감지용)")
    
    # 컨텍스트 파일들
    context_files: List[str] = Field(default_factory=list, description="컨텍스트 파일들")
    scripts: List[str] = Field(default_factory=list, description="스크립트 파일들")
    resources: List[str] = Field(default_factory=list, description="리소스 파일들")
    
    # 메타데이터
    version: str = Field(default="1.0.0")
    category: str = Field(default="general")
    tags: List[str] = Field(default_factory=list)
    dependencies: List[str] = Field(default_factory=list)
    
    # 로드된 컨텐츠
    instructions: str = Field(default="", description="스킬 지침")
    full_context: str = Field(default="", description="전체 컨텍스트")
    
    class Config:
        arbitrary_types_allowed = True


class SkillChangeEvent(BaseModel):
    """스킬 변경 이벤트."""
    skill_id: str
    change_type: SkillChangeType
    timestamp: datetime = Field(default_factory=datetime.now)
    old_hash: str = ""
    new_hash: str = ""
    details: str = ""


class FileWatcher:
    """파일 변경 감시기."""
    
    def __init__(
        self,
        watch_paths: List[Path],
        patterns: List[str] = None,
        callback: Optional[Callable] = None,
        poll_interval: float = 2.0
    ):
        self.watch_paths = watch_paths
        self.patterns = patterns or ["SKILL.md"]
        self.callback = callback
        self.poll_interval = poll_interval
        
        self._watching = False
        self._watch_task: Optional[asyncio.Task] = None
        self._file_states: Dict[str, str] = {}  # path -> hash
    
    async def start(self):
        """감시 시작."""
        if self._watching:
            return
        
        self._watching = True
        
        # 초기 상태 스냅샷
        self._file_states = await self._scan_files()
        
        self._watch_task = asyncio.create_task(self._watch_loop())
        logger.info(f"FileWatcher started: {len(self.watch_paths)} paths")
    
    async def stop(self):
        """감시 중지."""
        self._watching = False
        
        if self._watch_task:
            self._watch_task.cancel()
            try:
                await self._watch_task
            except asyncio.CancelledError:
                pass
        
        logger.info("FileWatcher stopped")
    
    async def _watch_loop(self):
        """감시 루프."""
        while self._watching:
            try:
                current_states = await self._scan_files()
                changes = self._detect_changes(current_states)
                
                if changes and self.callback:
                    await self._safe_callback(self.callback, changes)
                
                self._file_states = current_states
                
                await asyncio.sleep(self.poll_interval)
                
            except Exception as e:
                logger.error(f"Watch loop error: {e}")
                await asyncio.sleep(self.poll_interval)
    
    async def _scan_files(self) -> Dict[str, str]:
        """파일 스캔 및 해시 계산."""
        states = {}
        
        for watch_path in self.watch_paths:
            if not watch_path.exists():
                continue
            
            for pattern in self.patterns:
                for file_path in watch_path.rglob(pattern):
                    try:
                        content = file_path.read_text(encoding='utf-8')
                        file_hash = hashlib.md5(content.encode()).hexdigest()
                        states[str(file_path)] = file_hash
                    except Exception as e:
                        logger.debug(f"Failed to read {file_path}: {e}")
        
        return states
    
    def _detect_changes(
        self,
        current_states: Dict[str, str]
    ) -> List[SkillChangeEvent]:
        """변경 감지."""
        changes = []
        
        # 추가/수정 감지
        for path, new_hash in current_states.items():
            if path not in self._file_states:
                changes.append(SkillChangeEvent(
                    skill_id=self._path_to_skill_id(path),
                    change_type=SkillChangeType.ADDED,
                    new_hash=new_hash,
                    details=f"New skill discovered at {path}"
                ))
            elif self._file_states[path] != new_hash:
                changes.append(SkillChangeEvent(
                    skill_id=self._path_to_skill_id(path),
                    change_type=SkillChangeType.MODIFIED,
                    old_hash=self._file_states[path],
                    new_hash=new_hash,
                    details=f"Skill modified at {path}"
                ))
        
        # 삭제 감지
        for path in self._file_states:
            if path not in current_states:
                changes.append(SkillChangeEvent(
                    skill_id=self._path_to_skill_id(path),
                    change_type=SkillChangeType.REMOVED,
                    old_hash=self._file_states[path],
                    details=f"Skill removed from {path}"
                ))
        
        return changes
    
    def _path_to_skill_id(self, path: str) -> str:
        """경로에서 스킬 ID 추출."""
        path_obj = Path(path)
        # SKILL.md의 부모 디렉토리 이름을 스킬 ID로 사용
        return path_obj.parent.name
    
    async def _safe_callback(self, callback: Callable, *args, **kwargs):
        """안전한 콜백 실행."""
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(*args, **kwargs)
            else:
                callback(*args, **kwargs)
        except Exception as e:
            logger.warning(f"Callback error: {e}")


class SkillContextLoader:
    """
    스킬 컨텍스트 로더.
    
    ERA 스타일로 스킬 실행 시 전체 컨텍스트 주입.
    """
    
    def __init__(self):
        self.context_file_patterns = [
            "SKILL.md",
            "reference.md",
            "forms.md",
            "examples.md",
            "scripts/*.py",
            "resources/*"
        ]
    
    def load_full_context(
        self,
        skill_directory: Path,
        include_scripts: bool = True,
        include_resources: bool = False
    ) -> str:
        """
        스킬의 전체 컨텍스트 로드.
        
        Args:
            skill_directory: 스킬 디렉토리
            include_scripts: 스크립트 포함 여부
            include_resources: 리소스 포함 여부
        """
        context_parts = []
        
        # SKILL.md
        skill_md = skill_directory / "SKILL.md"
        if skill_md.exists():
            context_parts.append(f"# SKILL.md\n{skill_md.read_text(encoding='utf-8')}")
        
        # reference.md
        reference_md = skill_directory / "reference.md"
        if reference_md.exists():
            context_parts.append(f"\n# Reference\n{reference_md.read_text(encoding='utf-8')}")
        
        # forms.md
        forms_md = skill_directory / "forms.md"
        if forms_md.exists():
            context_parts.append(f"\n# Forms\n{forms_md.read_text(encoding='utf-8')}")
        
        # examples.md
        examples_md = skill_directory / "examples.md"
        if examples_md.exists():
            context_parts.append(f"\n# Examples\n{examples_md.read_text(encoding='utf-8')}")
        
        # 스크립트
        if include_scripts:
            scripts_dir = skill_directory / "scripts"
            if scripts_dir.exists():
                for script_file in scripts_dir.glob("*.py"):
                    try:
                        content = script_file.read_text(encoding='utf-8')
                        context_parts.append(
                            f"\n# Script: {script_file.name}\n```python\n{content}\n```"
                        )
                    except Exception as e:
                        logger.debug(f"Failed to load script {script_file}: {e}")
        
        # 리소스 (텍스트 파일만)
        if include_resources:
            resources_dir = skill_directory / "resources"
            if resources_dir.exists():
                for resource_file in resources_dir.iterdir():
                    if resource_file.is_file() and resource_file.suffix in ['.txt', '.md', '.json', '.yaml']:
                        try:
                            content = resource_file.read_text(encoding='utf-8')
                            context_parts.append(
                                f"\n# Resource: {resource_file.name}\n{content}"
                            )
                        except Exception as e:
                            logger.debug(f"Failed to load resource {resource_file}: {e}")
        
        return "\n\n".join(context_parts)
    
    def list_context_files(self, skill_directory: Path) -> Dict[str, List[str]]:
        """컨텍스트 파일 목록."""
        result = {
            "documents": [],
            "scripts": [],
            "resources": []
        }
        
        # 문서
        for doc_name in ["SKILL.md", "reference.md", "forms.md", "examples.md"]:
            doc_path = skill_directory / doc_name
            if doc_path.exists():
                result["documents"].append(doc_name)
        
        # 스크립트
        scripts_dir = skill_directory / "scripts"
        if scripts_dir.exists():
            result["scripts"] = [f.name for f in scripts_dir.glob("*.py")]
        
        # 리소스
        resources_dir = skill_directory / "resources"
        if resources_dir.exists():
            result["resources"] = [f.name for f in resources_dir.iterdir() if f.is_file()]
        
        return result


class SkillAutoDiscovery:
    """
    스킬 자동 발견 시스템.
    
    파일시스템에서 SKILL.md를 자동으로 스캔하고 핫 리로드 지원.
    """
    
    def __init__(
        self,
        search_paths: Optional[List[Path]] = None,
        enable_hot_reload: bool = True,
        poll_interval: float = 2.0
    ):
        # 검색 경로 설정
        if search_paths is None:
            # 기본 검색 경로
            project_root = Path(__file__).parent.parent.parent
            search_paths = [
                project_root / "skills",
                project_root / "src" / "core" / "skills",
                Path.home() / ".cursor" / "skills",
            ]
        
        self.search_paths = [p for p in search_paths if p.exists()]
        self.enable_hot_reload = enable_hot_reload
        self.poll_interval = poll_interval
        
        # 발견된 스킬
        self.skills: Dict[str, DiscoveredSkill] = {}
        
        # 컨텍스트 로더
        self.context_loader = SkillContextLoader()
        
        # 파일 감시기
        self.file_watcher: Optional[FileWatcher] = None
        
        # 콜백
        self.on_skill_discovered: Optional[Callable] = None
        self.on_skill_loaded: Optional[Callable] = None
        self.on_skill_reloaded: Optional[Callable] = None
        self.on_skill_removed: Optional[Callable] = None
        
        # 변경 이력
        self.change_history: List[SkillChangeEvent] = []
        
        logger.info(
            f"SkillAutoDiscovery initialized: {len(self.search_paths)} search paths, "
            f"hot_reload={enable_hot_reload}"
        )
    
    async def start(self):
        """자동 발견 시작."""
        # 초기 스캔
        await self.scan()
        
        # 핫 리로드 활성화
        if self.enable_hot_reload and self.search_paths:
            self.file_watcher = FileWatcher(
                watch_paths=self.search_paths,
                patterns=["SKILL.md"],
                callback=self._handle_file_changes,
                poll_interval=self.poll_interval
            )
            await self.file_watcher.start()
        
        logger.info(f"SkillAutoDiscovery started: {len(self.skills)} skills found")
    
    async def stop(self):
        """자동 발견 중지."""
        if self.file_watcher:
            await self.file_watcher.stop()
        
        logger.info("SkillAutoDiscovery stopped")
    
    async def scan(self) -> List[DiscoveredSkill]:
        """스킬 스캔."""
        discovered = []
        
        for search_path in self.search_paths:
            if not search_path.exists():
                continue
            
            for skill_md in search_path.rglob("SKILL.md"):
                try:
                    skill = await self._discover_skill(skill_md)
                    if skill:
                        self.skills[skill.skill_id] = skill
                        discovered.append(skill)
                        
                        if self.on_skill_discovered:
                            await self._safe_callback(self.on_skill_discovered, skill)
                
                except Exception as e:
                    logger.warning(f"Failed to discover skill at {skill_md}: {e}")
        
        return discovered
    
    async def _discover_skill(self, skill_md_path: Path) -> Optional[DiscoveredSkill]:
        """개별 스킬 발견."""
        skill_dir = skill_md_path.parent
        skill_id = skill_dir.name
        
        try:
            content = skill_md_path.read_text(encoding='utf-8')
            content_hash = hashlib.md5(content.encode()).hexdigest()
            
            # 메타데이터 파싱
            metadata = self._parse_metadata(content)
            
            # 컨텍스트 파일 목록
            context_files = self.context_loader.list_context_files(skill_dir)
            
            skill = DiscoveredSkill(
                skill_id=skill_id,
                name=metadata.get("name", skill_id),
                description=metadata.get("description", ""),
                path=str(skill_md_path),
                directory=str(skill_dir),
                status=SkillStatus.DISCOVERED,
                last_modified=datetime.fromtimestamp(skill_md_path.stat().st_mtime),
                content_hash=content_hash,
                context_files=context_files.get("documents", []),
                scripts=context_files.get("scripts", []),
                resources=context_files.get("resources", []),
                version=metadata.get("version", "1.0.0"),
                category=metadata.get("category", "general"),
                tags=metadata.get("tags", []),
                dependencies=metadata.get("dependencies", []),
                instructions=self._extract_instructions(content)
            )
            
            return skill
            
        except Exception as e:
            logger.error(f"Failed to parse skill at {skill_md_path}: {e}")
            return None
    
    async def load_skill(
        self,
        skill_id: str,
        include_full_context: bool = True
    ) -> Optional[DiscoveredSkill]:
        """스킬 로드."""
        if skill_id not in self.skills:
            logger.warning(f"Skill {skill_id} not found")
            return None
        
        skill = self.skills[skill_id]
        skill.status = SkillStatus.LOADING
        
        try:
            if include_full_context:
                skill.full_context = self.context_loader.load_full_context(
                    Path(skill.directory),
                    include_scripts=True,
                    include_resources=False
                )
            
            skill.status = SkillStatus.LOADED
            
            if self.on_skill_loaded:
                await self._safe_callback(self.on_skill_loaded, skill)
            
            logger.info(f"Loaded skill: {skill_id}")
            return skill
            
        except Exception as e:
            skill.status = SkillStatus.FAILED
            logger.error(f"Failed to load skill {skill_id}: {e}")
            return None
    
    async def reload_skill(self, skill_id: str) -> Optional[DiscoveredSkill]:
        """스킬 리로드."""
        if skill_id not in self.skills:
            return None
        
        old_skill = self.skills[skill_id]
        old_skill.status = SkillStatus.RELOADING
        
        skill_md_path = Path(old_skill.path)
        
        if not skill_md_path.exists():
            self.skills.pop(skill_id, None)
            return None
        
        new_skill = await self._discover_skill(skill_md_path)
        
        if new_skill:
            new_skill = await self.load_skill(skill_id, include_full_context=True)
            
            if self.on_skill_reloaded:
                await self._safe_callback(self.on_skill_reloaded, new_skill)
            
            logger.info(f"Reloaded skill: {skill_id}")
        
        return new_skill
    
    async def _handle_file_changes(self, changes: List[SkillChangeEvent]):
        """파일 변경 처리."""
        for change in changes:
            self.change_history.append(change)
            
            if change.change_type == SkillChangeType.ADDED:
                logger.info(f"New skill detected: {change.skill_id}")
                # 새 스킬 발견 시 스캔
                await self.scan()
            
            elif change.change_type == SkillChangeType.MODIFIED:
                logger.info(f"Skill modified: {change.skill_id}")
                # 수정된 스킬 리로드
                await self.reload_skill(change.skill_id)
            
            elif change.change_type == SkillChangeType.REMOVED:
                logger.info(f"Skill removed: {change.skill_id}")
                # 제거된 스킬 정리
                if change.skill_id in self.skills:
                    removed_skill = self.skills.pop(change.skill_id)
                    
                    if self.on_skill_removed:
                        await self._safe_callback(self.on_skill_removed, removed_skill)
    
    def _parse_metadata(self, content: str) -> Dict[str, Any]:
        """SKILL.md에서 메타데이터 파싱."""
        import re
        
        metadata = {}
        
        # YAML frontmatter 파싱 (--- ... ---)
        yaml_match = re.match(r'^---\s*\n(.*?)\n---', content, re.DOTALL)
        if yaml_match:
            yaml_content = yaml_match.group(1)
            for line in yaml_content.split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # 리스트 처리
                    if value.startswith('[') and value.endswith(']'):
                        value = [v.strip().strip('"\'') for v in value[1:-1].split(',')]
                    
                    metadata[key] = value
        
        # 제목에서 이름 추출
        title_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
        if title_match and 'name' not in metadata:
            metadata['name'] = title_match.group(1).strip()
        
        # 설명 추출 (첫 번째 단락)
        if 'description' not in metadata:
            # 첫 번째 빈 줄 이전의 텍스트 (헤더 제외)
            lines = content.split('\n')
            desc_lines = []
            for line in lines:
                if not line.strip():
                    if desc_lines:
                        break
                elif not line.startswith('#'):
                    desc_lines.append(line.strip())
            
            if desc_lines:
                metadata['description'] = ' '.join(desc_lines)[:200]
        
        return metadata
    
    def _extract_instructions(self, content: str) -> str:
        """지침 섹션 추출."""
        import re
        
        # ## Instructions 또는 ## How to Use 섹션 찾기
        patterns = [
            r'##\s+Instructions?\s*\n(.*?)(?=\n##|\Z)',
            r'##\s+How to Use\s*\n(.*?)(?=\n##|\Z)',
            r'##\s+Usage\s*\n(.*?)(?=\n##|\Z)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # 전체 내용 반환 (섹션 없는 경우)
        return content
    
    def get_skill(self, skill_id: str) -> Optional[DiscoveredSkill]:
        """스킬 조회."""
        return self.skills.get(skill_id)
    
    def get_skills_by_category(self, category: str) -> List[DiscoveredSkill]:
        """카테고리별 스킬 조회."""
        return [
            skill for skill in self.skills.values()
            if skill.category == category
        ]
    
    def get_skills_by_tag(self, tag: str) -> List[DiscoveredSkill]:
        """태그별 스킬 조회."""
        return [
            skill for skill in self.skills.values()
            if tag in skill.tags
        ]
    
    def search_skills(
        self,
        query: str,
        top_k: int = 5
    ) -> List[DiscoveredSkill]:
        """스킬 검색."""
        query_lower = query.lower()
        
        results = []
        for skill in self.skills.values():
            score = 0
            
            # 이름 매칭
            if query_lower in skill.name.lower():
                score += 3
            
            # 설명 매칭
            if query_lower in skill.description.lower():
                score += 2
            
            # 태그 매칭
            for tag in skill.tags:
                if query_lower in tag.lower():
                    score += 1
            
            if score > 0:
                results.append((skill, score))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return [skill for skill, _ in results[:top_k]]
    
    def get_statistics(self) -> Dict[str, Any]:
        """통계 반환."""
        category_counts = {}
        for skill in self.skills.values():
            category_counts[skill.category] = category_counts.get(skill.category, 0) + 1
        
        status_counts = {}
        for skill in self.skills.values():
            status_counts[skill.status.value] = status_counts.get(skill.status.value, 0) + 1
        
        return {
            "total_skills": len(self.skills),
            "search_paths": [str(p) for p in self.search_paths],
            "categories": category_counts,
            "status_distribution": status_counts,
            "hot_reload_enabled": self.enable_hot_reload,
            "recent_changes": len([
                c for c in self.change_history
                if (datetime.now() - c.timestamp).total_seconds() < 3600
            ])
        }
    
    async def _safe_callback(self, callback: Callable, *args, **kwargs):
        """안전한 콜백 실행."""
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(*args, **kwargs)
            else:
                callback(*args, **kwargs)
        except Exception as e:
            logger.warning(f"Callback error: {e}")


# Singleton instance
_skill_auto_discovery: Optional[SkillAutoDiscovery] = None


def get_skill_auto_discovery(
    search_paths: Optional[List[Path]] = None,
    enable_hot_reload: bool = True
) -> SkillAutoDiscovery:
    """SkillAutoDiscovery 싱글톤 인스턴스 반환."""
    global _skill_auto_discovery
    
    if _skill_auto_discovery is None:
        _skill_auto_discovery = SkillAutoDiscovery(
            search_paths=search_paths,
            enable_hot_reload=enable_hot_reload
        )
    
    return _skill_auto_discovery
