"""Forge Master Execution Personas - Dispatched-goal behavior directives

외부 CLI 에이전트에 작업을 위임하기 전, 목표 텍스트 앞단에 주입하는
실행 페르소나 지시문 레지스트리. Ponytail(미니멀 코드/과잉설계 방지)과
Caveman(극단적 토큰 압축) 두 페르소나를 결합한 Blacksmith를 포함한다.
"""

from typing import Dict, Optional

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

BLACKSMITH_DIRECTIVE = (
    f"{PONYTAIL_DIRECTIVE} {CAVEMAN_DIRECTIVE} Balance both: correctness and minimal "
    "footprint come first, token compression second - never sacrifice a working fix for a "
    "shorter one."
)

EXECUTION_PERSONAS: Dict[str, str] = {
    "ponytail": PONYTAIL_DIRECTIVE,
    "caveman": CAVEMAN_DIRECTIVE,
    "blacksmith": BLACKSMITH_DIRECTIVE,
}


def apply_persona(goal_text: str, persona: Optional[str]) -> str:
    """Prepend an execution persona directive to a dispatched goal string.

    Unknown or absent persona names are a no-op.
    """
    directive = EXECUTION_PERSONAS.get(persona) if persona else None
    if not directive:
        return goal_text
    return f"[Persona: {persona}] {directive}\n\n{goal_text}"
