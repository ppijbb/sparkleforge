"""Plugin discovery: scan configs/plugins and skills/*/.sparkleforge-plugin for plugin.json."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.core.plugin_system.manifest import PLUGIN_MANIFEST_DIR, PluginManifest

logger = logging.getLogger(__name__)


@dataclass
class PluginInfo:
    """Discovered plugin: root path and manifest."""

    root: Path
    manifest: PluginManifest
    skill_ids: List[str] = field(default_factory=list)


class PluginDiscovery:
    """Discover all plugins and build a merged skill registry from manifests + optional skills_registry."""

    def __init__(self, project_root: Path) -> None:
        self.project_root = Path(project_root)
        self.configs_plugins_dir = self.project_root / "configs" / "plugins"
        self.skills_dir = self.project_root / "skills"
        self._plugins: List[PluginInfo] = []
        self._registry_skills: Dict[str, Dict[str, Any]] = {}
        self._registry_categories: Dict[str, List[str]] = {}
        self._registry_dependencies: Dict[str, List[str]] = {}

    def discover(self) -> List[PluginInfo]:
        """Scan configs/plugins and skills for .sparkleforge-plugin/plugin.json; return PluginInfo list."""
        self._plugins = []
        # configs/plugins/<name>/
        if self.configs_plugins_dir.is_dir():
            for child in self.configs_plugins_dir.iterdir():
                if child.is_dir():
                    manifest = PluginManifest.load(child)
                    if manifest:
                        skill_ids = list(manifest.skills) if manifest.skills else []
                        self._plugins.append(
                            PluginInfo(root=child, manifest=manifest, skill_ids=skill_ids)
                        )
        # skills/<skill_id>/.sparkleforge-plugin/plugin.json (plugin declares single skill = dir name)
        if self.skills_dir.is_dir():
            for skill_dir in self.skills_dir.iterdir():
                if not skill_dir.is_dir():
                    continue
                manifest = PluginManifest.load(skill_dir)
                if manifest:
                    skill_id = skill_dir.name
                    skill_ids = list(manifest.skills) if manifest.skills else [skill_id]
                    self._plugins.append(
                        PluginInfo(root=skill_dir, manifest=manifest, skill_ids=skill_ids)
                    )
        logger.info("Discovered %d plugin(s)", len(self._plugins))
        return self._plugins

    def get_plugin_roots(self) -> List[Path]:
        """Return list of plugin roots for HookRunner."""
        if not self._plugins:
            self.discover()
        return [p.root for p in self._plugins]

    def build_registry_from_plugins(
        self,
        existing_skills: Optional[Dict[str, Dict[str, Any]]] = None,
        existing_categories: Optional[Dict[str, List[str]]] = None,
        existing_dependencies: Optional[Dict[str, List[str]]] = None,
    ) -> tuple[Dict[str, Dict[str, Any]], Dict[str, List[str]], Dict[str, List[str]]]:
        """Build skills dict, skill_categories, skill_dependencies from plugins + existing (merge).

        Plugins can declare skills[] (skill_ids). For each skill_id we keep existing entry if any,
        else we do not invent metadata here (SkillManager still uses skills_registry.json for
        full metadata and SKILL.md for load). So this only adds plugin roots; registry content
        still comes from skills_registry.json. We merge: plugin-discovered skill IDs are
        registered as enabled; existing_* overwritten by plugin declarations where applicable.
        """
        skills = dict(existing_skills or {})
        skill_categories = dict(existing_categories or {})
        skill_dependencies = dict(existing_dependencies or {})

        if not self._plugins:
            self.discover()

        for pinfo in self._plugins:
            for skill_id in pinfo.skill_ids:
                if skill_id not in skills:
                    skills[skill_id] = {
                        "skill_id": skill_id,
                        "name": skill_id.replace("_", " ").title(),
                        "description": pinfo.manifest.description or "",
                        "version": pinfo.manifest.version,
                        "category": "general",
                        "tags": [],
                        "path": f"skills/{skill_id}" if (self.skills_dir / skill_id).exists() else "",
                        "enabled": True,
                        "dependencies": [],
                        "required_tools": [],
                        "author": pinfo.manifest.author.name if pinfo.manifest.author else "Unknown",
                        "created_at": "",
                        "updated_at": "",
                        "metadata": {"capabilities": [], "compatibility": {}},
                    }
                if skill_id not in skill_categories.get("general", []):
                    cat = skill_categories.setdefault("general", [])
                    if skill_id not in cat:
                        cat.append(skill_id)

        self._registry_skills = skills
        self._registry_categories = skill_categories
        self._registry_dependencies = skill_dependencies
        return skills, skill_categories, skill_dependencies
