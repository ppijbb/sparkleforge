"""Forge Master Execution Personas - Dispatched-goal behavior directives.

외부 CLI 에이전트에 작업을 위임하기 전, 목표 텍스트 앞단에 주입하는 실행
페르소나 지시문 레지스트리.

Registry pattern
----------------
``EXECUTION_PERSONAS`` is a module-level ``Dict[str, str]`` mapping a persona
key to its directive text, mirroring the same pattern already used by
``RED_TEAM_PERSONAS`` (``src/core/adversarial_council.py``) and
``PERSONA_PERSPECTIVES`` (``src/agents/creativity_agent.py``).

To add a new persona:
1. Define a ``<NAME>_DIRECTIVE`` module constant describing the behavior.
2. Register it under a lowercase key in ``EXECUTION_PERSONAS``.
3. Callers select it via ``execute_task_with_master_control(persona=...)``;
   unknown names are a no-op in ``apply_persona``.

Selection rules
---------------
``persona`` is selectable per-call via ``execute_task_with_master_control``.
Automatic inference from ``task_query``/``required_capabilities`` is left to
``ForgeMasterRouter`` and is intentionally not wired here so the registry
stays a pure data table.
"""

from typing import Dict, List, Optional

PONYTAIL_DIRECTIVE = (
    "Act like the laziest senior engineer in the room: ship the smallest correct diff, "
    "never add abstractions, helpers, or config for hypothetical future needs, and prefer "
    "deleting code over adding it when both solve the task."
)

CAVEMAN_DIRECTIVE = (
    "Compress all reasoning and prose to the essential minimum: no preamble, no restating "
    "the task, no step-by-step narration, terse fragments over full sentences where meaning "
    "survives."
)

MENTOR_DIRECTIVE = (
    "Act like a patient senior mentor: optimize for reader comprehension over "
    "brevity. Explain the why behind non-obvious changes, surface assumptions, "
    "and annotate trade-offs so onboarding docs, RFCs, and reviews stay "
    "self-contained for future readers."
)

SENTINEL_DIRECTIVE = (
    "Act like a security-hardened engineer: default to defensive coding at every "
    "trust boundary. Validate and sanitize all inputs, prefer allow-lists over "
    "deny-lists, fail closed, avoid secret leakage in logs/errors, and call out "
    "privilege, injection, and unsafe-deserialization risks before shipping."
)

RACECAR_DIRECTIVE = (
    "Act like a performance-first engineer: optimize for runtime and memory over "
    "code brevity. Prefer efficient data structures and algorithms, avoid "
    "unnecessary allocations/copies, and call out hot paths, complexity cliffs, "
    "and measurable regressions before shipping."
)

BLACKSMITH_DIRECTIVE = (
    f"{PONYTAIL_DIRECTIVE} {CAVEMAN_DIRECTIVE} Balance both: correctness and minimal "
    "footprint come first, token compression second - never sacrifice a working fix for a "
    "shorter one."
)

EXECUTION_PERSONAS: Dict[str, str] = {
    "ponytail": PONYTAIL_DIRECTIVE,
    "caveman": CAVEMAN_DIRECTIVE,
    "blacksmith": BLACKSMITH_DIRECTIVE,
    "mentor": MENTOR_DIRECTIVE,
    "sentinel": SENTINEL_DIRECTIVE,
    "racecar": RACECAR_DIRECTIVE,
}


def list_personas() -> List[str]:
    """Return the registered persona keys (stable order for callers/UI)."""
    return list(EXECUTION_PERSONAS.keys())


def apply_persona(goal_text: str, persona: Optional[str]) -> str:
    """Prepend an execution persona directive to a dispatched goal string.

    Unknown or absent persona names are a no-op.
    """
    directive = EXECUTION_PERSONAS.get(persona) if persona else None
    if not directive:
        return goal_text
    return f"[Persona: {persona}] {directive}\n\n{goal_text}"
