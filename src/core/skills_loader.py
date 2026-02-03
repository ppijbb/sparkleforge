"""
Skills Loader - Load and parse individual Skills

SKILL.md 파일을 파싱하고 스크립트 및 리소스를 로드하는 모듈.
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class SkillMetadata:
    """Skill 메타데이터."""
    skill_id: str
    name: str
    description: str
    version: str
    category: str
    tags: List[str]
    author: str
    created_at: str
    updated_at: str
    path: str
    enabled: bool = True
    dependencies: List[str] = field(default_factory=list)
    required_tools: List[str] = field(default_factory=list)
    capabilities: List[str] = field(default_factory=list)


@dataclass
class Skill:
    """로드된 Skill 객체."""
    metadata: SkillMetadata
    instructions: str
    overview: str
    usage: str
    scripts: List[str] = field(default_factory=list)
    resources: List[str] = field(default_factory=list)
    skill_path: Path = None
    
    def __post_init__(self):
        """스크립트 및 리소스 경로 완성."""
        if self.skill_path:
            # scripts 디렉토리의 파일들
            scripts_dir = self.skill_path / "scripts"
            if scripts_dir.exists():
                self.scripts = [str(f.relative_to(self.skill_path)) 
                              for f in scripts_dir.iterdir() 
                              if f.is_file() and f.suffix == ".py"]
            
            # resources 디렉토리의 파일들
            resources_dir = self.skill_path / "resources"
            if resources_dir.exists():
                self.resources = [str(f.relative_to(self.skill_path)) 
                                 for f in resources_dir.iterdir() 
                                 if f.is_file()]


class SkillLoader:
    """개별 Skill을 로드하는 클래스."""
    
    def __init__(self, project_root: Optional[Path] = None):
        """초기화."""
        if project_root is None:
            # 기본값: 현재 파일 기준으로 프로젝트 루트 찾기
            project_root = Path(__file__).parent.parent.parent
        
        self.project_root = Path(project_root)
        self.skills_dir = self.project_root / "skills"
    
    def load_skill(self, skill_id: str) -> Optional[Skill]:
        """Skill을 로드."""
        skill_path = self.skills_dir / skill_id
        
        if not skill_path.exists():
            logger.error(f"Skill directory not found: {skill_path}")
            return None
        
        skill_md_path = skill_path / "SKILL.md"
        if not skill_md_path.exists():
            logger.error(f"SKILL.md not found for skill: {skill_id}")
            return None
        
        try:
            # SKILL.md 파싱
            skill_content = skill_md_path.read_text(encoding='utf-8')
            parsed = self._parse_skill_md(skill_content)
            
            # 메타데이터 추출
            metadata = self._extract_metadata(skill_id, skill_path, parsed)
            
            # Skill 객체 생성
            skill = Skill(
                metadata=metadata,
                instructions=parsed.get("instructions", ""),
                overview=parsed.get("overview", ""),
                usage=parsed.get("usage", ""),
                skill_path=skill_path
            )
            
            logger.info(f"✅ Loaded skill: {skill_id} v{metadata.version}")
            return skill
            
        except Exception as e:
            logger.error(f"Failed to load skill {skill_id}: {e}")
            return None
    
    def _parse_skill_md(self, content: str) -> Dict[str, Any]:
        """SKILL.md 파일 파싱."""
        parsed = {
            "overview": "",
            "instructions": "",
            "usage": "",
            "dependencies": [],
            "capabilities": [],
            "metadata": {}
        }
        
        # 섹션 분리
        sections = self._split_sections(content)
        
        # Overview 추출
        if "## Overview" in sections:
            parsed["overview"] = sections["## Overview"].strip()
        
        # Instructions 추출
        if "## Instructions" in sections:
            parsed["instructions"] = sections["## Instructions"].strip()
        
        # Usage 추출
        if "## Usage" in sections:
            parsed["usage"] = sections["## Usage"].strip()
        
        # Capabilities 추출
        if "## Capabilities" in sections:
            capabilities_text = sections["## Capabilities"]
            # List 형태로 파싱 (- 로 시작하는 리스트 아이템)
            capabilities = []
            for line in capabilities_text.split('\n'):
                if line.strip().startswith('-'):
                    capabilities.append(line.strip()[1:].strip())
            parsed["capabilities"] = capabilities
        
        # Dependencies 추출
        if "## Dependencies" in sections:
            deps_text = sections["## Dependencies"]
            deps = []
            for line in deps_text.split('\n'):
                if line.strip().startswith('-'):
                    deps.append(line.strip()[1:].strip())
            parsed["dependencies"] = deps
        
        # Metadata 추출 (JSON 블록)
        metadata_match = re.search(r'```json\s*(\{.*?\})\s*```', content, re.DOTALL)
        if metadata_match:
            try:
                parsed["metadata"] = json.loads(metadata_match.group(1))
            except json.JSONDecodeError:
                logger.warning("Failed to parse metadata JSON")
        
        return parsed
    
    def _split_sections(self, content: str) -> Dict[str, str]:
        """마크다운을 섹션별로 분리."""
        sections = {}
        current_section = None
        current_content = []
        
        for line in content.split('\n'):
            if line.startswith('##'):
                # 이전 섹션 저장
                if current_section:
                    sections[current_section] = '\n'.join(current_content).strip()
                
                # 새 섹션 시작
                current_section = line.strip()
                current_content = []
            else:
                if current_section:
                    current_content.append(line)
        
        # 마지막 섹션 저장
        if current_section:
            sections[current_section] = '\n'.join(current_content).strip()
        
        return sections
    
    def _extract_metadata(self, skill_id: str, skill_path: Path, parsed: Dict[str, Any]) -> SkillMetadata:
        """SkillMetadata 객체 생성."""
        # 메타데이터는 parsed["metadata"]와 skills_registry.json에서 가져옴
        metadata_json = parsed.get("metadata", {})
        
        # skills_registry.json도 확인
        registry_path = self.project_root / "skills_registry.json"
        registry_metadata = {}
        if registry_path.exists():
            try:
                with open(registry_path, 'r', encoding='utf-8') as f:
                    registry = json.load(f)
                    if "skills" in registry and skill_id in registry["skills"]:
                        registry_metadata = registry["skills"][skill_id]
            except Exception as e:
                logger.warning(f"Failed to load registry metadata: {e}")
        
        # 우선순위: parsed metadata > registry metadata > 기본값
        return SkillMetadata(
            skill_id=metadata_json.get("skill_id") or registry_metadata.get("skill_id") or skill_id,
            name=registry_metadata.get("name", skill_id.replace("_", " ").title()),
            description=registry_metadata.get("description", ""),
            version=metadata_json.get("version") or registry_metadata.get("version", "1.0.0"),
            category=metadata_json.get("category") or registry_metadata.get("category", "general"),
            tags=metadata_json.get("tags") or registry_metadata.get("tags", []),
            author=metadata_json.get("author") or registry_metadata.get("author", "Unknown"),
            created_at=metadata_json.get("created_at") or registry_metadata.get("created_at", datetime.now().isoformat()),
            updated_at=metadata_json.get("updated_at") or registry_metadata.get("updated_at", datetime.now().isoformat()),
            path=str(skill_path.relative_to(self.project_root)),
            enabled=registry_metadata.get("enabled", True),
            dependencies=registry_metadata.get("dependencies", parsed.get("dependencies", [])),
            required_tools=registry_metadata.get("required_tools", []),
            capabilities=registry_metadata.get("metadata", {}).get("capabilities", parsed.get("capabilities", []))
        )
    
    def load_skill_scripts(self, skill: Skill) -> Dict[str, Any]:
        """Skill의 스크립트들을 로드."""
        scripts = {}
        
        for script_path in skill.scripts:
            full_path = skill.skill_path / script_path
            if full_path.exists():
                try:
                    # Python 스크립트는 import 가능하도록 처리
                    scripts[script_path] = {
                        "path": str(full_path),
                        "loaded": False,  # 실제 import는 필요할 때 수행
                        "type": "python"
                    }
                except Exception as e:
                    logger.warning(f"Failed to load script {script_path}: {e}")
        
        return scripts
    
    def load_skill_resources(self, skill: Skill) -> Dict[str, Any]:
        """Skill의 리소스들을 로드."""
        resources = {}
        
        for resource_path in skill.resources:
            full_path = skill.skill_path / resource_path
            if full_path.exists():
                try:
                    # 리소스는 파일 내용 또는 경로 저장
                    if full_path.suffix in ['.json', '.yaml', '.yml', '.txt']:
                        resources[resource_path] = {
                            "path": str(full_path),
                            "content": full_path.read_text(encoding='utf-8'),
                            "type": full_path.suffix[1:]  # .json -> json
                        }
                    else:
                        resources[resource_path] = {
                            "path": str(full_path),
                            "type": "file"
                        }
                except Exception as e:
                    logger.warning(f"Failed to load resource {resource_path}: {e}")
        
        return resources

