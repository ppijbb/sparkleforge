"""Plugin discovery: scan configs/plugins, skills/*/.sparkleforge-plugin, SKILL.md packages, and MCP servers."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from src.core.plugin_system.manifest import PluginManifest

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
        self._open_skills: Dict[str, Dict[str, Any]] = {}

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
        
        # Discover open SKILL.md packages and MCP-to-Skills integration if skills_dir exists
        if self.skills_dir.is_dir():
            for skill_dir in self.skills_dir.iterdir():
                if not skill_dir.is_dir():
                    continue
                skill_md = skill_dir / "SKILL.md"
                if skill_md.is_file():
                    metadata = self._parse_skill_md_frontmatter(skill_md)
                    skill_id = metadata.get("name") or skill_dir.name
                    self._open_skills[skill_id] = {
                        "skill_id": skill_id,
                        "name": metadata.get("name", skill_id.replace("_", " ").title()),
                        "description": metadata.get("description", ""),
                        "version": metadata.get("version", "1.0.0"),
                        "category": metadata.get("category", "general"),
                        "tags": metadata.get("tags", []),
                        "path": str(skill_dir.relative_to(self.project_root)),
                        "enabled": True,
                        "lazy": True,
                        "body_path": str(skill_md),
                        "metadata": metadata,
                    }

        # Expose MCP server prompts/skills as server:skill if available via registry or manager
        self._discover_mcp_skills()

        logger.info("Discovered %d plugin(s)", len(self._plugins))
        return self._plugins

    def _parse_skill_md_frontmatter(self, skill_md_path: Path) -> Dict[str, Any]:
        """Lazy-load YAML/TOML frontmatter from SKILL.md for lightweight token budget metadata."""
        try:
            content = skill_md_path.read_text(encoding="utf-8")
            if content.startswith("---"):
                parts = content.split("---", 2)
                if len(parts) >= 3:
                    frontmatter_raw = parts[1].strip()
                    meta: Dict[str, Any] = {}
                    for line in frontmatter_raw.splitlines():
                        if ":" in line:
                            k, v = line.split(":", 1)
                            meta[k.strip()] = v.strip().strip("'\"")
                    return meta
        except Exception as e:
            logger.warning("Failed to parse SKILL.md frontmatter from %s: %s", skill_md_path, e)
        return {}

    def _discover_mcp_skills(self) -> None:
        """Expose connected MCP server tools/prompts as server:skill format."""
        try:
            from src.core.mcp_integration.hub import get_mcp_hub
            hub = get_mcp_hub()
            if hub and hasattr(hub, "list_tools"):
                for tool in hub.list_tools():
                    server_name = getattr(tool, "server_name", "mcp")
                    tool_name = getattr(tool, "name", str(tool))
                    skill_id = f"{server_name}:{tool_name}"
                    self._open_skills[skill_id] = {
                        "skill_id": skill_id,
                        "name": tool_name.replace("_", " ").title(),
                        "description": getattr(tool, "description", ""),
                        "category": "mcp",
                        "enabled": True,
                    }
        except Exception:
            pass

    def get_plugin_roots(self) -> List[Path]:
        """Return list of plugin roots for HookRunner."""
        if not self._plugins:
            self.discover()
        return [p.root for p in self._plugins]

    def build_registry_from_plugins(
        self,
        existing_skills: Dict[str, Dict[str, Any]] | None = None,
        existing_categories: Dict[str, List[str]] | None = None,
        existing_dependencies: Dict[str, List[str]] | None = None,
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
                        "path": (
                            f"skills/{skill_id}" if (self.skills_dir / skill_id).exists() else ""
                        ),
                        "enabled": True,
                        "dependencies": [],
                        "required_tools": [],
                        "author": (
                            pinfo.manifest.author.name if pinfo.manifest.author else "Unknown"
                        ),
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
        
        # Merge open SKILL.md and MCP skills into registry
        if not self._open_skills:
            self.discover()
        for sid, sdata in self._open_skills.items():
            if sid not in skills:
                skills[sid] = sdata
            cat = skill_categories.setdefault(sdata.get("category", "general"), [])
            if sid not in cat:
                cat.append(sid)

        return skills, skill_categories, skill_dependencies
