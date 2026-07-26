"""Multi-Agent Adversarial Red-Team Council.

A dedicated adversarial council ("Red Team") that rigorously critiques
synthesized ideas for hidden logical fallacies, edge-case vulnerabilities,
and unstated assumptions.
"""

import asyncio
import json
import logging
import re
from typing import Any, Dict, List, Optional

from src.core.llm_council import CouncilError, query_model_via_openrouter

logger = logging.getLogger(__name__)


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


async def run_red_team_council(
    user_query: str,
    stage3_result: Dict[str, Any],
    council_models: List[str],
    api_key: str,
    api_url: str,
    timeout: float = 120.0,
    personas: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """Run the multi-agent adversarial red-team council against a synthesis.

    Args:
        user_query: The original user question.
        stage3_result: The Chairman's synthesized answer (``{"model", "response"}``).
        council_models: Models to assign as red-team critics.
        api_key: OpenRouter API key.
        api_url: OpenRouter API URL.
        timeout: Request timeout in seconds.
        personas: Optional override of the red-team personas.

    Returns:
        A dictionary with per-persona critiques, aggregated risk, and a
        consolidated summary of the red-team verdict.
    """
    personas = personas or RED_TEAM_PERSONAS
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
    }
