"""Explicit startup stages for SparkleForge runtime initialization."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable


@dataclass(frozen=True)
class BootstrapStage:
    """One startup stage in the bootstrap graph."""

    name: str
    fn: Callable[[], Awaitable[dict[str, Any]]]
    depends_on: tuple[str, ...] = ()
    critical: bool = True


@dataclass
class BootstrapStageResult:
    """Recorded result for one startup stage."""

    name: str
    ok: bool
    duration_ms: float
    payload: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass
class BootstrapResult:
    """Aggregate bootstrap output."""

    ok: bool
    stages: list[BootstrapStageResult]
    values: dict[str, Any]

    def render_lines(self) -> list[str]:
        """Render a human-readable stage summary."""
        lines: list[str] = []
        for stage in self.stages:
            status = "OK" if stage.ok else "FAIL"
            detail = f" {stage.error}" if stage.error else ""
            lines.append(
                f"[{status}] {stage.name} ({stage.duration_ms:.1f} ms){detail}"
            )
        return lines


class BootstrapGraph:
    """Minimal explicit startup DAG used by main entrypoints and debug flows."""

    def __init__(self, project_root: Path | None = None, runtime_mode: str = "local"):
        self.project_root = Path(project_root or Path.cwd())
        self.runtime_mode = runtime_mode

    def _default_stages(self) -> list[BootstrapStage]:
        return [
            BootstrapStage("config", self._stage_config),
            BootstrapStage("database", self._stage_database, depends_on=("config",)),
            BootstrapStage("mcp_hub", self._stage_mcp_hub, depends_on=("config",)),
            BootstrapStage(
                "skills_plugins_hooks",
                self._stage_skills_plugins_hooks,
                depends_on=("config",),
            ),
            BootstrapStage("trust_gate", self._stage_trust_gate, depends_on=("config",)),
            BootstrapStage(
                "runtime_mode",
                self._stage_runtime_mode,
                depends_on=("trust_gate",),
            ),
        ]

    async def _stage_config(self) -> dict[str, Any]:
        from src.core.researcher_config import load_config_from_env

        config = load_config_from_env()
        return {
            "provider": getattr(getattr(config, "llm", None), "provider", "unknown"),
            "project_root": str(self.project_root),
        }

    async def _stage_database(self) -> dict[str, Any]:
        from src.core.db.database_driver import get_database_driver, set_database_driver
        from src.core.db.sqlite_driver import SQLiteDriver

        driver = get_database_driver()
        if driver is None:
            sqlite_db_path = self.project_root / "data" / "sparkleforge.db"
            sqlite_db_path.parent.mkdir(parents=True, exist_ok=True)
            driver = SQLiteDriver(str(sqlite_db_path))
            set_database_driver(driver)
        return {"driver": driver.__class__.__name__}

    async def _stage_mcp_hub(self) -> dict[str, Any]:
        from src.core.mcp_integration import get_mcp_hub

        hub = get_mcp_hub()
        return {"registered_tools": len(hub.registry.get_all_tool_names())}

    async def _stage_skills_plugins_hooks(self) -> dict[str, Any]:
        from src.core.skills_manager import get_skill_manager

        skill_manager = get_skill_manager()
        hook_runner = skill_manager.get_hook_runner()
        plugin_roots = getattr(hook_runner, "plugin_roots", []) if hook_runner else []
        return {
            "skills": len(skill_manager.get_all_skills(enabled_only=True)),
            "plugin_roots": len(plugin_roots),
            "hooks_enabled": hook_runner is not None,
        }

    async def _stage_trust_gate(self) -> dict[str, Any]:
        from src.core.trust_gate import TrustGate

        trust = await TrustGate(
            project_root=self.project_root,
            runtime_mode=self.runtime_mode,
        ).evaluate()
        return {
            "level": trust.level.value,
            "deny_names": sorted(trust.deny_names),
            "deny_prefixes": list(trust.deny_prefixes),
            "allowed_mcp_servers": sorted(trust.allowed_mcp_servers or []),
        }

    async def _stage_runtime_mode(self) -> dict[str, Any]:
        return {"mode": self.runtime_mode}

    async def run(self) -> BootstrapResult:
        """Run the startup graph sequentially and collect stage diagnostics."""
        stage_results: list[BootstrapStageResult] = []
        values: dict[str, Any] = {}
        completed: set[str] = set()

        for stage in self._default_stages():
            missing = [dep for dep in stage.depends_on if dep not in completed]
            if missing:
                result = BootstrapStageResult(
                    name=stage.name,
                    ok=False,
                    duration_ms=0.0,
                    error=f"missing dependencies: {', '.join(missing)}",
                )
                stage_results.append(result)
                if stage.critical:
                    return BootstrapResult(ok=False, stages=stage_results, values=values)
                continue

            started = time.perf_counter()
            try:
                payload = await stage.fn()
                duration_ms = (time.perf_counter() - started) * 1000
                result = BootstrapStageResult(
                    name=stage.name,
                    ok=True,
                    duration_ms=duration_ms,
                    payload=payload,
                )
                values[stage.name] = payload
                completed.add(stage.name)
            except Exception as e:
                duration_ms = (time.perf_counter() - started) * 1000
                result = BootstrapStageResult(
                    name=stage.name,
                    ok=False,
                    duration_ms=duration_ms,
                    error=str(e),
                )
                stage_results.append(result)
                if stage.critical:
                    return BootstrapResult(ok=False, stages=stage_results, values=values)
                continue

            stage_results.append(result)

        return BootstrapResult(ok=True, stages=stage_results, values=values)
