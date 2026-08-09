"""Multi-Agent Adversarial Red-Team Council.

A multi-round aggressive debate engine where specialized sub-agents attack
core assumptions, expose hidden flaws, and push solutions beyond conventional
limits. A dedicated adversarial council ("Red Team") rigorously critiques
synthesized ideas for hidden logical fallacies, edge-case vulnerabilities, and
unstated assumptions across multiple dialectic rounds.
"""

import asyncio
import json
import logging
import re
from typing import Any, Dict, List, Optional

from src.core.llm_council import CouncilError, query_model_via_openrouter

logger = logging.getLogger(__name__)


RED_TEAM_DEBATE_PERSONAS = [
    {
        "name": "AssaultSquad",
        "focus": (
            "aggressively attack the core assumptions of the synthesis, expose "
            "hidden flaws, and force the solution beyond conventional limits"
        ),
    },
    {
        "name": "DevilsAdvocate",
        "focus": (
            "argue the strongest possible opposing position, invert the "
            "synthesis's conclusions, and surface the weakest load-bearing claims"
        ),
    },
    {
        "name": "StressTester",
        "focus": (
            "apply extreme dialectic pressure by constructing adversarial "
            "counterexamples and worst-case scenarios the synthesis cannot survive"
        ),
    },
]


def _build_rebuttal_prompt(
    user_query: str,
    synthesis_text: str,
    prior_rounds: List[Dict[str, Any]],
    persona: Dict[str, str],
    round_number: int,
) -> str:
    """Build the multi-round rebuttal prompt for an aggressive debate persona."""
    prior_summary = ""
    if prior_rounds:
        prior_summary = "\n\nPrior debate rounds:\n" + "\n\n".join(
            f"Round {entry.get('round', idx + 1)} - {entry.get('persona', 'unknown')}:\n"
            f"{entry.get('summary', entry.get('raw', ''))}"
            for idx, entry in enumerate(prior_rounds)
        )
    return f"""You are a member of an aggressive adversarial debate council conducting round {round_number} of a multi-round extreme dialectic argumentation protocol.

Your persona: {persona["name"]}
Your sole focus: {persona["focus"]}

Original question:
 {user_query}

Synthesized answer under attack:
{synthesis_text}
{prior_summary}

Your task this round:
1. Attack the core assumptions that survived the previous round.
2. Expose any hidden flaws the prior rebuttals missed.
3. Push the solution beyond conventional limits with a concrete stronger alternative.

Respond as STRICT JSON with this schema:
{{
  "persona": "{persona["name"]}",
  "round": {round_number},
  "findings": [
    {{
      "severity": "critical|high|medium|low",
      "flaw": "short name of the flaw",
      "location": "where in the synthesis or prior round it occurs",
      "explanation": "why this is a problem",
      "mitigation": "concrete fix or stronger alternative"
    }}
  ],
  "overall_risk": "critical|high|medium|low|none",
  "summary": "one paragraph summary of this round's dialectic verdict"
}}

Only output the JSON object. Do not include any prose outside the JSON."""


RED_TEAM_PERSONAS = [
    {
        "name": "LogicalFallacyHunter",
        "focus": (
            "hidden logical fallacies, circular reasoning, false dichotomies, "
            "non sequiturs, and unstated premises in the synthesized answer"
        ),
    },
    {
        "name": "EdgeCaseAuditor",
        "focus": (
            "edge-case vulnerabilities, boundary conditions, empty/null inputs, "
            "concurrency hazards, and failure modes the synthesis ignores"
        ),
    },
    {
        "name": "AssumptionInterrogator",
        "focus": (
            "unstated assumptions, implicit dependencies, hidden preconditions, "
            "and claims that require evidence the synthesis does not provide"
        ),
    },
]


def _build_critique_prompt(
    user_query: str,
    synthesis_text: str,
    persona: Dict[str, str],
) -> str:
    """Build the red-team critique prompt for a single adversarial persona."""
    return f"""You are a member of an adversarial Red Team council reviewing a synthesized answer.

Your persona: {persona["name"]}
Your sole focus: {persona["focus"]}

Original question:
{user_query}

Synthesized answer under review:
{synthesis_text}

Your task:
1. Identify the most serious problems in the synthesized answer within your focus area.
2. For each problem, state the specific flaw, where it occurs, and why it matters.
3. Propose a concrete mitigation or additional check that would address the flaw.

Respond as STRICT JSON with this schema:
{{
  "persona": "{persona["name"]}",
  "findings": [
    {{
      "severity": "critical|high|medium|low",
      "flaw": "short name of the flaw",
      "location": "where in the synthesis it occurs",
      "explanation": "why this is a problem",
      "mitigation": "concrete fix or additional check"
    }}
  ],
  "overall_risk": "critical|high|medium|low|none",
  "summary": "one paragraph summary of the red-team verdict"
}}

If you find no problems within your focus area, return an empty findings list with overall_risk "none".
Only output the JSON object. Do not include any prose outside the JSON."""


def _parse_critique_response(text: str, persona_name: str) -> Dict[str, Any]:
    """Parse a red-team critique response, tolerating minor formatting drift."""
    if not text:
        return {
            "persona": persona_name,
            "findings": [],
            "overall_risk": "none",
            "summary": "Red-team model returned an empty response.",
            "raw": text,
        }

    # Try to extract a JSON object from the response.
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {
            "persona": persona_name,
            "findings": [],
            "overall_risk": "unknown",
            "summary": "Red-team response was not parseable as JSON.",
            "raw": text,
        }

    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        return {
            "persona": persona_name,
            "findings": [],
            "overall_risk": "unknown",
            "summary": f"Red-team response failed JSON parsing: {exc}",
            "raw": text,
        }

    parsed.setdefault("persona", persona_name)
    parsed.setdefault("findings", [])
    parsed.setdefault("overall_risk", "unknown")
    parsed.setdefault("summary", "")
    parsed["raw"] = text
    return parsed


async def _query_persona(
    persona: Dict[str, str],
    user_query: str,
    synthesis_text: str,
    model: str,
    api_key: str,
    api_url: str,
    timeout: float,
) -> Dict[str, Any]:
    """Query a single red-team persona model and parse its critique."""
    prompt = _build_critique_prompt(user_query, synthesis_text, persona)
    messages = [{"role": "user", "content": prompt}]

    try:
        response = await query_model_via_openrouter(model, messages, api_key, api_url, timeout)
    except CouncilError as exc:
        logger.warning("Red-team persona %s model %s failed: %s", persona["name"], model, exc)
        return {
            "persona": persona["name"],
            "model": model,
            "findings": [],
            "overall_risk": "unknown",
            "summary": f"Red-team model query failed: {exc}",
            "raw": "",
        }

    text = response.get("content", "")
    critique = _parse_critique_response(text, persona["name"])
    critique["model"] = model
    return critique


def _aggregate_risk(risks: List[str]) -> str:
    """Aggregate per-persona risk levels into an overall council risk."""
    order = {"critical": 4, "high": 3, "medium": 2, "low": 1, "none": 0}
    if not risks:
        return "none"
    best = max(risks, key=lambda r: order.get(r, 0))
    return best if best in order else "unknown"


async def _query_debate_persona(
    persona: Dict[str, str],
    user_query: str,
    synthesis_text: str,
    prior_rounds: List[Dict[str, Any]],
    round_number: int,
    model: str,
    api_key: str,
    api_url: str,
    timeout: float,
) -> Dict[str, Any]:
    """Query a single debate persona for a given dialectic round."""
    prompt = _build_rebuttal_prompt(user_query, synthesis_text, prior_rounds, persona, round_number)
    messages = [{"role": "user", "content": prompt}]

    try:
        response = await query_model_via_openrouter(model, messages, api_key, api_url, timeout)
    except CouncilError as exc:
        logger.warning(
            "Debate persona %s model %s failed in round %d: %s",
            persona["name"],
            model,
            round_number,
            exc,
        )
        return {
            "persona": persona["name"],
            "model": model,
            "round": round_number,
            "findings": [],
            "overall_risk": "unknown",
            "summary": f"Debate model query failed: {exc}",
            "raw": "",
        }

    text = response.get("content", "")
    critique = _parse_critique_response(text, persona["name"])
    critique["model"] = model
    critique["round"] = round_number
    return critique


async def _run_debate_round(
    user_query: str,
    synthesis_text: str,
    prior_rounds: List[Dict[str, Any]],
    round_number: int,
    council_models: List[str],
    api_key: str,
    api_url: str,
    timeout: float,
    personas: List[Dict[str, str]],
) -> List[Dict[str, Any]]:
    """Run a single multi-round aggressive debate round across personas."""
    assignments = [
        (persona, council_models[i % len(council_models)])
        for i, persona in enumerate(personas)
    ]

    tasks = [
        _query_debate_persona(
            persona,
            user_query,
            synthesis_text,
            prior_rounds,
            round_number,
            model,
            api_key,
            api_url,
            timeout,
        )
        for persona, model in assignments
    ]

    critiques = await asyncio.gather(*tasks, return_exceptions=True)

    round_critiques: List[Dict[str, Any]] = []
    for (persona, model), critique in zip(assignments, critiques):
        if isinstance(critique, Exception):
            logger.warning(
                "Debate persona %s (model %s) raised in round %d: %s",
                persona["name"],
                model,
                round_number,
                critique,
            )
            round_critiques.append(
                {
                    "persona": persona["name"],
                    "model": model,
                    "round": round_number,
                    "findings": [],
                    "overall_risk": "unknown",
                    "summary": f"Debate persona failed: {critique}",
                    "raw": "",
                }
            )
        else:
            round_critiques.append(critique)
    return round_critiques


async def run_red_team_council(
    user_query: str,
    stage3_result: Dict[str, Any],
    council_models: List[str],
    api_key: str,
    api_url: str,
    timeout: float = 120.0,
    personas: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    rounds: int = 1,
    debate_personas: Optional[List[Dict[str, str]]] = None,
    """Run the multi-agent adversarial red-team council against a synthesis.

    Args:
        user_query: The original user question.
        stage3_result: The Chairman's synthesized answer (``{"model", "response"}``).
        council_models: Models to assign as red-team critics.
        api_key: OpenRouter API key.
        api_url: OpenRouter API URL.
        timeout: Request timeout in seconds.
        personas: Optional override of the red-team personas.
        rounds: Number of aggressive dialectic debate rounds (>=1).
        debate_personas: Optional override of the multi-round debate personas.

    Returns:
        A dictionary with per-persona critiques, aggregated risk, and a
        consolidated summary of the red-team verdict.
    """
    personas = personas or RED_TEAM_PERSONAS
    debate_personas = debate_personas or RED_TEAM_DEBATE_PERSONAS
    synthesis_text = stage3_result.get("response", "") if stage3_result else ""

    if not synthesis_text:
        logger.warning("Red-team council invoked with an empty synthesis; skipping critique.")
        return {
            "personas": [],
            "overall_risk": "none",
            "summary": "No synthesis was provided for red-team critique.",
            "synthesis_model": stage3_result.get("model") if stage3_result else None,
        }

    if not council_models:
        logger.warning("Red-team council has no models available; skipping critique.")
        return {
            "personas": [],
            "overall_risk": "none",
            "summary": "No red-team models were configured.",
            "synthesis_model": stage3_result.get("model") if stage3_result else None,
        }

    try:
        rounds = max(1, int(rounds))
    except (TypeError, ValueError):
        rounds = 1

    # Assign one model per persona, cycling through the available models so
    # every persona gets a distinct critic when possible.
    assignments = [
        (persona, council_models[i % len(council_models)])
        for i, persona in enumerate(personas)
    ]

    tasks = [
        _query_persona(persona, user_query, synthesis_text, model, api_key, api_url, timeout)
        for persona, model in assignments
    ]

    critiques = await asyncio.gather(*tasks, return_exceptions=True)

    parsed_critiques: List[Dict[str, Any]] = []
    for (persona, model), critique in zip(assignments, critiques):
        if isinstance(critique, Exception):
            logger.warning(
                "Red-team persona %s (model %s) raised: %s",
                persona["name"],
                model,
                critique,
            )
            parsed_critiques.append(
                {
                    "persona": persona["name"],
                    "model": model,
                    "findings": [],
                    "overall_risk": "unknown",
                    "summary": f"Red-team persona failed: {critique}",
                    "raw": "",
                }
            )
        else:
            parsed_critiques.append(critique)

    overall_risk = _aggregate_risk([c.get("overall_risk", "unknown") for c in parsed_critiques])

    all_rounds: List[Dict[str, Any]] = []
    prior_rounds: List[Dict[str, Any]] = list(parsed_critiques)
    if rounds > 1:
        for round_number in range(2, rounds + 1):
            round_critiques = await _run_debate_round(
                user_query,
                synthesis_text,
                prior_rounds,
                round_number,
                council_models,
                api_key,
                api_url,
                timeout,
                debate_personas,
            )
            parsed_critiques.extend(round_critiques)
            all_rounds.append(
                {
                    "round": round_number,
                    "critiques": round_critiques,
                    "overall_risk": _aggregate_risk(
                        [c.get("overall_risk", "unknown") for c in round_critiques]
                    ),
                }
            )
            prior_rounds = round_critiques
            overall_risk = _aggregate_risk(
                [overall_risk]
                + [c.get("overall_risk", "unknown") for c in round_critiques]
            )

    all_findings = [
        {"persona": c.get("persona"), **finding}
        for c in parsed_critiques
        for finding in c.get("findings", [])
    ]

    summary = " ".join(c.get("summary", "") for c in parsed_critiques).strip()
    if not summary:
        summary = "Red-team council produced no summary."

    logger.info(
        "Red-team council completed: %d critiques, overall risk=%s",
        len(parsed_critiques),
        overall_risk,
    )

    return {
        "personas": parsed_critiques,
        "overall_risk": overall_risk,
        "findings": all_findings,
        "summary": summary,
        "synthesis_model": stage3_result.get("model") if stage3_result else None,
        "rounds": rounds,
        "debate_rounds": all_rounds,
    }
