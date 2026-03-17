"""Plugin manifest (plugin.json) schema and loading.

Compatible with Claude Code plugin manifest style; SparkleForge uses
.sparkleforge-plugin/plugin.json per plugin root.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

PLUGIN_MANIFEST_DIR = ".sparkleforge-plugin"
PLUGIN_MANIFEST_FILE = "plugin.json"

# kebab-case plugin name validation
PLUGIN_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9]*(-[a-z0-9]+)*$")


@dataclass
class PluginAuthor:
    """Plugin author (name, email, url)."""

    name: str
    email: Optional[str] = None
    url: Optional[str] = None


@dataclass
class PluginManifest:
    """Plugin manifest loaded from plugin.json."""

    name: str
    version: str
    description: str = ""
    author: Optional[PluginAuthor] = None
    homepage: Optional[str] = None
    skills: List[str] = field(default_factory=list)
    commands: List[str] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, plugin_root: Path) -> Optional["PluginManifest"]:
        """Load plugin.json from plugin_root / .sparkleforge-plugin / plugin.json."""
        manifest_path = plugin_root / PLUGIN_MANIFEST_DIR / PLUGIN_MANIFEST_FILE
        if not manifest_path.is_file():
            return None
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to load plugin manifest %s: %s", manifest_path, e)
            return None
        return cls.from_dict(data, plugin_root)

    @classmethod
    def from_dict(
        cls, data: Dict[str, Any], plugin_root: Optional[Path] = None
    ) -> Optional["PluginManifest"]:
        """Build PluginManifest from dict (e.g. from plugin.json)."""
        name = (data.get("name") or "").strip()
        if not name:
            logger.warning("Plugin manifest missing required 'name'")
            return None
        if not PLUGIN_NAME_PATTERN.match(name):
            logger.warning("Plugin name must be kebab-case: %s", name)

        version = data.get("version") or "0.1.0"
        description = (data.get("description") or "").strip()

        author = None
        if "author" in data:
            a = data["author"]
            if isinstance(a, dict):
                author = PluginAuthor(
                    name=(a.get("name") or "").strip() or "Unknown",
                    email=a.get("email"),
                    url=a.get("url"),
                )
            elif isinstance(a, str):
                author = PluginAuthor(name=a.strip() or "Unknown")

        skills = list(data.get("skills") or [])
        commands = list(data.get("commands") or [])

        return cls(
            name=name,
            version=version,
            description=description,
            author=author,
            homepage=data.get("homepage"),
            skills=skills,
            commands=commands,
            raw=data,
        )
