"""Skill Creator - Interactive Skill creation tool

대화형으로 Skills를 생성하는 도구.
사용자에게 Skills 구조를 질문하고 SKILL.md 및 관련 파일을 자동 생성.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class SkillCreator:
    """Skills 생성 도구."""

    def __init__(self, project_root: Path | None = None):
        """초기화."""
        if project_root is None:
            project_root = Path(__file__).parent.parent.parent

        self.project_root = Path(project_root)
        self.skills_dir = self.project_root / "skills"
        self.skills_dir.mkdir(parents=True, exist_ok=True)

    def create_skill_interactive(self) -> str | None:
        """대화형으로 Skill 생성."""
        print("\n🔨 Skill Creator - Interactive Mode")
        print("=" * 60)

        # 1. 기본 정보 입력
        skill_id = (
            input("\nSkill ID (lowercase, underscore): ")
            .strip()
            .lower()
            .replace(" ", "_")
        )
        if not skill_id:
            print("❌ Skill ID is required")
            return None

        skill_name = input("Skill Name: ").strip() or skill_id.replace("_", " ").title()
        description = input("Description: ").strip()
        category = (
            input("Category (planning/execution/evaluation/synthesis): ").strip()
            or "general"
        )

        # 2. Capabilities 입력
        print("\nEnter capabilities (one per line, empty to finish):")
        capabilities = []
        while True:
            cap = input("  - ").strip()
            if not cap:
                break
            capabilities.append(cap)

        # 3. Instructions 입력
        print("\nEnter instructions (multi-line, 'END' to finish):")
        instructions = []
        while True:
            line = input()
            if line.strip() == "END":
                break
            instructions.append(line)
        instructions_text = "\n".join(instructions)

        # 4. Dependencies 입력
        print("\nEnter dependency skill IDs (comma-separated, empty if none):")
        deps_input = input("  Dependencies: ").strip()
        dependencies = (
            [d.strip() for d in deps_input.split(",") if d.strip()]
            if deps_input
            else []
        )

        # 5. Tags 입력
        print("\nEnter tags (comma-separated):")
        tags_input = input("  Tags: ").strip()
        tags = (
            [t.strip() for t in tags_input.split(",") if t.strip()]
            if tags_input
            else []
        )

        # 6. Skill 생성
        skill_path = self.create_skill(
            skill_id=skill_id,
            name=skill_name,
            description=description,
            category=category,
            capabilities=capabilities,
            instructions=instructions_text,
            dependencies=dependencies,
            tags=tags,
        )

        if skill_path:
            print(f"\n✅ Skill created successfully at: {skill_path}")
            return skill_id
        else:
            print("\n❌ Failed to create skill")
            return None

    def create_skill(
        self,
        skill_id: str,
        name: str,
        description: str,
        category: str,
        capabilities: List[str],
        instructions: str,
        dependencies: List[str] = None,
        tags: List[str] = None,
        author: str = "Local Researcher Team",
    ) -> Path | None:
        """Skill 생성."""
        skill_dir = self.skills_dir / skill_id
        skill_dir.mkdir(parents=True, exist_ok=True)

        # scripts, resources 디렉토리 생성
        (skill_dir / "scripts").mkdir(exist_ok=True)
        (skill_dir / "resources").mkdir(exist_ok=True)

        # SKILL.md 생성
        skill_md = self._generate_skill_md(
            skill_id=skill_id,
            name=name,
            description=description,
            category=category,
            capabilities=capabilities,
            instructions=instructions,
            dependencies=dependencies or [],
            tags=tags or [],
            author=author,
        )

        skill_md_path = skill_dir / "SKILL.md"
        skill_md_path.write_text(skill_md, encoding="utf-8")

        logger.info(f"✅ Created skill: {skill_id}")
        return skill_dir

    def _generate_skill_md(
        self,
        skill_id: str,
        name: str,
        description: str,
        category: str,
        capabilities: List[str],
        instructions: str,
        dependencies: List[str],
        tags: List[str],
        author: str,
    ) -> str:
        """SKILL.md 템플릿 생성."""
        capabilities_text = "\n".join([f"- {cap}" for cap in capabilities])
        dependencies_text = (
            "\n".join([f"- {dep}" for dep in dependencies])
            if dependencies
            else "- None"
        )

        metadata_json = json.dumps(
            {
                "skill_id": skill_id,
                "version": "1.0.0",
                "category": category,
                "tags": tags,
                "author": author,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
            },
            indent=2,
        )

        md_template = f"""# {name}

## Overview
{description}

## Capabilities
{capabilities_text}

## Instructions

{instructions}

## Usage

This skill is automatically invoked when relevant tasks are detected.

## Dependencies
{dependencies_text}

## Resources
- Templates and guidelines stored in `resources/` directory

## Scripts
- Utility scripts stored in `scripts/` directory

## Metadata
```json
{metadata_json}
```
"""
        return md_template

    def update_registry(self, skill_id: str, skill_data: Dict[str, Any]):
        """configs/skills_registry.json 또는 skills_registry.json 업데이트."""
        registry_path = self.project_root / "configs" / "skills_registry.json"
        if not registry_path.exists():
            registry_path = self.project_root / "skills_registry.json"

        if registry_path.exists():
            with open(registry_path, encoding="utf-8") as f:
                registry = json.load(f)
        else:
            registry = {
                "version": "1.0.0",
                "registry_updated_at": datetime.now().isoformat(),
                "skills": {},
                "skill_categories": {},
                "skill_dependencies": {},
            }

        # Skills 추가/업데이트
        registry["skills"][skill_id] = skill_data
        registry["registry_updated_at"] = datetime.now().isoformat()

        # 카테고리 업데이트
        category = skill_data.get("category", "general")
        if category not in registry["skill_categories"]:
            registry["skill_categories"][category] = []
        if skill_id not in registry["skill_categories"][category]:
            registry["skill_categories"][category].append(skill_id)

        # 의존성 업데이트
        dependencies = skill_data.get("dependencies", [])
        if dependencies:
            registry["skill_dependencies"][skill_id] = dependencies

        # 저장
        registry_path.parent.mkdir(parents=True, exist_ok=True)
        with open(registry_path, "w", encoding="utf-8") as f:
            json.dump(registry, f, indent=2, ensure_ascii=False)

        logger.info(f"✅ Updated registry for skill: {skill_id}")


def main():
    """CLI 진입점."""
    creator = SkillCreator()
    skill_id = creator.create_skill_interactive()

    if skill_id:
        print(f"\n🎉 Skill '{skill_id}' created successfully!")
        print("   Edit SKILL.md to customize further.")
    else:
        print("\n❌ Skill creation cancelled or failed")


if __name__ == "__main__":
    main()
