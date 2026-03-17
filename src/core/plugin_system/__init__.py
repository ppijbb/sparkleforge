"""SparkleForge plugin system: dynamic plugin discovery, manifest (plugin.json), and lifecycle hooks.

Plugins are discovered from:
- configs/plugins/ (each subdir with .sparkleforge-plugin/plugin.json)
- skills/<skill_id>/.sparkleforge-plugin/ (optional per-skill plugin manifest)

Hooks (hooks.json) support: PreTaskRun, PostTaskRun, PreToolUse, PostToolUse.
"""

from src.core.plugin_system.manifest import PluginManifest
from src.core.plugin_system.discovery import PluginDiscovery, PluginInfo
from src.core.plugin_system.hooks import HookRunner, HookPhase

__all__ = [
    "PluginManifest",
    "PluginDiscovery",
    "PluginInfo",
    "HookRunner",
    "HookPhase",
]
