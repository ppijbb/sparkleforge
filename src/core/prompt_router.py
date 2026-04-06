"""Lightweight prompt routing for CLI/runtime entrypoints."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Iterable

from src.core.execution_registry import (
    ExecutionRegistry,
    RegisteredCommand,
    RegisteredSkill,
    RegisteredTool,
    RegisteredTrigger,
)
from src.core.trust_gate import TrustContext


class RouteTargetType(str, Enum):
    """Kinds of routable runtime targets."""

    COMMAND = "command"
    SKILL = "skill"
    TOOL = "tool"
    AUTOMATION = "automation"


@dataclass(frozen=True)
class RouteScore:
    """Scored route candidate."""

    target: str
    target_type: RouteTargetType
    score: float
    matched_tokens: tuple[str, ...]
    source: str


class PromptRouter:
    """Route natural-language prompts to the best matching runtime target."""

    def _tokenize(self, text: str) -> tuple[str, ...]:
        return tuple(re.findall(r"[a-z0-9_:+.-]+", (text or "").lower()))

    def _score_tokens(
        self,
        prompt_tokens: tuple[str, ...],
        candidate_tokens: Iterable[str],
    ) -> tuple[float, tuple[str, ...]]:
        candidate_tokens = tuple(token for token in candidate_tokens if token)
        if not prompt_tokens or not candidate_tokens:
            return 0.0, ()

        prompt_set = set(prompt_tokens)
        matched = tuple(token for token in candidate_tokens if token in prompt_set)
        if not matched:
            return 0.0, ()

        score = float(len(matched))
        if set(candidate_tokens).issubset(prompt_set):
            score += 1.5
        return score, matched

    def _command_candidate_tokens(self, command: RegisteredCommand) -> tuple[str, ...]:
        parts = list(self._tokenize(command.name))
        for alias in command.aliases:
            parts.extend(self._tokenize(alias))
        for hint in command.hints:
            parts.extend(self._tokenize(hint))
        parts.extend(self._tokenize(command.description))
        return tuple(parts)

    async def route(
        self,
        prompt: str,
        pool: ExecutionRegistry,
        trust: TrustContext,
    ) -> list[RouteScore]:
        """Return the top route candidates for a prompt."""
        registry = pool.filter_by_trust(trust)
        prompt_tokens = self._tokenize(prompt)
        results: list[RouteScore] = []

        for command in registry.commands:
            score, matched = self._score_tokens(
                prompt_tokens,
                self._command_candidate_tokens(command),
            )
            if score > 0:
                results.append(
                    RouteScore(
                        target=command.name,
                        target_type=RouteTargetType.COMMAND,
                        score=score + 2.0,
                        matched_tokens=matched,
                        source="builtin",
                    )
                )

        for skill in registry.skills:
            score, matched = self._score_tokens(
                prompt_tokens,
                self._tokenize(" ".join((skill.skill_id, skill.name, skill.description, " ".join(skill.tags)))),
            )
            if score > 0:
                results.append(
                    RouteScore(
                        target=skill.skill_id,
                        target_type=RouteTargetType.SKILL,
                        score=score + 1.0,
                        matched_tokens=matched,
                        source="skill",
                    )
                )

        for tool in registry.tools:
            score, matched = self._score_tokens(
                prompt_tokens,
                self._tokenize(" ".join((tool.name, tool.description, tool.source))),
            )
            if score > 0:
                results.append(
                    RouteScore(
                        target=tool.name,
                        target_type=RouteTargetType.TOOL,
                        score=score,
                        matched_tokens=matched,
                        source=tool.source,
                    )
                )

        for trigger in registry.triggers:
            score, matched = self._score_tokens(
                prompt_tokens,
                self._tokenize(
                    " ".join(
                        (trigger.name, trigger.schedule_id, trigger.cron_expression, trigger.description)
                    )
                ),
            )
            if score > 0:
                results.append(
                    RouteScore(
                        target=trigger.schedule_id,
                        target_type=RouteTargetType.AUTOMATION,
                        score=score,
                        matched_tokens=matched,
                        source="schedule",
                    )
                )

        return sorted(results, key=lambda item: item.score, reverse=True)[:5]
