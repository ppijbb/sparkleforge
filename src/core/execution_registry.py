"""Registry of executable commands, skills, tools, and schedule triggers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.core.trust_gate import TrustContext


@dataclass(frozen=True)
class RegisteredCommand:
    """One callable CLI/REPL command."""

    name: str
    dispatch: tuple[str, ...]
    description: str
    aliases: tuple[str, ...] = ()
    hints: tuple[str, ...] = ()
    requires_args: bool = False


@dataclass(frozen=True)
class RegisteredSkill:
    """One skill available to the runtime."""

    skill_id: str
    name: str
    description: str
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class RegisteredTool:
    """One tool available to the runtime."""

    name: str
    description: str
    source: str
    mcp_server: str | None = None


@dataclass(frozen=True)
class RegisteredTrigger:
    """One automation/schedule trigger."""

    name: str
    description: str
    schedule_id: str
    cron_expression: str


@dataclass(frozen=True)
class ExecutionRegistry:
    """Single source of truth for routable runtime targets."""

    commands: tuple[RegisteredCommand, ...]
    skills: tuple[RegisteredSkill, ...]
    tools: tuple[RegisteredTool, ...]
    triggers: tuple[RegisteredTrigger, ...]

    def lookup(self, name: str) -> Any | None:
        """Look up a registrant by canonical name or alias."""
        name = (name or "").strip().lower()
        if not name:
            return None

        for command in self.commands:
            if name == command.name.lower() or name in {alias.lower() for alias in command.aliases}:
                return command

        for skill in self.skills:
            if name in {skill.skill_id.lower(), skill.name.lower()}:
                return skill

        for tool in self.tools:
            if name == tool.name.lower():
                return tool

        for trigger in self.triggers:
            if name in {trigger.name.lower(), trigger.schedule_id.lower()}:
                return trigger

        return None

    def filter_by_trust(self, trust: TrustContext) -> ExecutionRegistry:
        """Filter tool entries according to trust policy."""
        tools = tuple(tool for tool in self.tools if trust.allows_tool(tool.name, tool.mcp_server))
        return ExecutionRegistry(
            commands=self.commands,
            skills=self.skills,
            tools=tools,
            triggers=self.triggers,
        )

    @classmethod
    async def build(
        cls,
        *,
        mcp_hub: Any,
        skill_manager: Any,
        scheduler: Any,
        trust: TrustContext,
        commands: tuple[RegisteredCommand, ...] | None = None,
    ) -> ExecutionRegistry:
        """Build the runtime registry from current managers."""
        commands = commands or default_command_registry()

        tools = []
        for tool_name in mcp_hub.get_allowed_tools(trust):
            info = mcp_hub.registry.get_tool_info(tool_name)
            tools.append(
                RegisteredTool(
                    name=tool_name,
                    description=info.description if info else "",
                    source="mcp" if info and info.mcp_server else "local",
                    mcp_server=info.mcp_server if info else None,
                )
            )

        skills = tuple(
            RegisteredSkill(
                skill_id=meta.skill_id,
                name=meta.name,
                description=meta.description,
                tags=tuple(meta.tags),
            )
            for meta in skill_manager.get_all_skills(enabled_only=True)
        )

        triggers = tuple(
            RegisteredTrigger(
                name=schedule.name,
                description=schedule.user_query,
                schedule_id=schedule.schedule_id,
                cron_expression=schedule.cron_expression,
            )
            for schedule in scheduler.list_schedules()
        )

        return cls(
            commands=tuple(commands),
            skills=skills,
            tools=tuple(tools),
            triggers=triggers,
        )


def default_command_registry() -> tuple[RegisteredCommand, ...]:
    """Canonical command catalog for routing and discovery."""
    return (
        RegisteredCommand(
            name="help",
            dispatch=("help",),
            description="Show help and command usage",
            hints=("help", "usage", "commands"),
        ),
        RegisteredCommand(
            name="schedule list",
            dispatch=("schedule", "list"),
            description="List all schedules",
            hints=("schedule", "list", "cron", "jobs"),
        ),
        RegisteredCommand(
            name="schedule stats",
            dispatch=("schedule", "stats"),
            description="Show scheduler statistics",
            hints=("schedule", "stats", "cron"),
        ),
        RegisteredCommand(
            name="session list",
            dispatch=("session", "list"),
            description="List active sessions",
            hints=("session", "list"),
        ),
        RegisteredCommand(
            name="session stats",
            dispatch=("session", "stats"),
            description="Show session statistics",
            hints=("session", "stats"),
        ),
        RegisteredCommand(
            name="context show",
            dispatch=("context", "show"),
            description="Show current project context",
            hints=("context", "show", "project"),
        ),
        RegisteredCommand(
            name="context reload",
            dispatch=("context", "reload"),
            description="Reload project context",
            hints=("context", "reload"),
        ),
        RegisteredCommand(
            name="checkpoint list",
            dispatch=("checkpoint", "list"),
            description="List saved checkpoints",
            hints=("checkpoint", "list"),
        ),
        RegisteredCommand(
            name="schedule create",
            dispatch=("schedule", "create"),
            description="Create a cron schedule",
            hints=("schedule", "create", "cron"),
            aliases=("schedule add",),
            requires_args=True,
        ),
        RegisteredCommand(
            name="schedule run",
            dispatch=("schedule", "run"),
            description="Run a schedule immediately",
            hints=("schedule", "run", "execute"),
            requires_args=True,
        ),
    )
