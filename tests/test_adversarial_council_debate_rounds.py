"""Issue #974: the multi-round debate params were spliced in after the
function signature already closed, so `rounds` and `debate_personas` became
a dead tuple-assignment statement in the function body instead of real
parameters -- run_red_team_council(..., rounds=2) raised TypeError, and the
docstring stopped being a docstring. Also covers two related bugs found in
the same review: round-1 critiques mislabeled by list position instead of
round number, and the "unknown" risk being silently outranked by "none" on
a tie.
"""

import json
import sys
import types

import pytest

# src.core.adversarial_council and src.core.llm_council import each other at
# module level (pre-existing on main, unrelated to this fix -- reproduces
# with `git stash` too). A fresh interpreter can't import either in
# isolation; the real app only survives it because something else happens
# to import one of them first through a longer, order-dependent chain. Stub
# llm_council's two symbols so this test module doesn't depend on that.
if "src.core.llm_council" not in sys.modules:
    _stub = types.ModuleType("src.core.llm_council")

    class CouncilError(Exception):
        pass

    async def query_model_via_openrouter(*args, **kwargs):
        raise NotImplementedError("stub: patched per-test")

    _stub.CouncilError = CouncilError
    _stub.query_model_via_openrouter = query_model_via_openrouter
    sys.modules["src.core.llm_council"] = _stub

from src.core.adversarial_council import _aggregate_risk, run_red_team_council


def _json_response(persona_name: str, round_number: int, overall_risk: str = "low") -> dict:
    return {
        "content": json.dumps(
            {
                "persona": persona_name,
                "round": round_number,
                "findings": [],
                "overall_risk": overall_risk,
                "summary": f"{persona_name} round {round_number} summary",
            }
        )
    }


def test_aggregate_risk_prefers_unknown_over_none_on_tie():
    assert _aggregate_risk(["none", "unknown", "unknown"]) == "unknown"
    assert _aggregate_risk(["unknown", "none"]) == "unknown"


def test_aggregate_risk_still_prefers_real_risk_over_unknown():
    assert _aggregate_risk(["unknown", "low"]) == "low"
    assert _aggregate_risk(["unknown", "critical"]) == "critical"


@pytest.mark.asyncio
async def test_run_red_team_council_accepts_rounds_kwarg(monkeypatch):
    """The core regression: rounds/debate_personas must be real, callable
    keyword parameters, not dead code after the signature."""
    calls = []

    async def fake_query(model, messages, api_key, api_url, timeout):
        calls.append(messages[0]["content"])
        # Alternate persona name isn't tracked here; just return a generic
        # valid critique so parsing succeeds regardless of which persona/round.
        return {"content": json.dumps(
            {"findings": [], "overall_risk": "low", "summary": "ok"}
        )}

    monkeypatch.setattr(
        "src.core.adversarial_council.query_model_via_openrouter", fake_query
    )

    result = await run_red_team_council(
        user_query="q",
        stage3_result={"response": "synthesis", "model": "m"},
        council_models=["model-a"],
        api_key="key",
        api_url="url",
        rounds=2,
    )

    assert result["rounds"] == 2
    assert len(result["debate_rounds"]) == 1
    assert result["debate_rounds"][0]["round"] == 2
    assert calls, "expected the mocked model to have been invoked"


@pytest.mark.asyncio
async def test_round_one_critiques_are_labeled_round_one_not_by_list_position(monkeypatch):
    prompts_seen = []

    async def fake_query(model, messages, api_key, api_url, timeout):
        prompt = messages[0]["content"]
        prompts_seen.append(prompt)
        if "round 2" in prompt.lower():
            return {"content": json.dumps(
                {"findings": [], "overall_risk": "low", "summary": "round 2 reply"}
            )}
        return {"content": json.dumps(
            {"findings": [], "overall_risk": "low", "summary": "round 1 reply"}
        )}

    monkeypatch.setattr(
        "src.core.adversarial_council.query_model_via_openrouter", fake_query
    )

    await run_red_team_council(
        user_query="q",
        stage3_result={"response": "synthesis", "model": "m"},
        council_models=["model-a", "model-b", "model-c"],
        api_key="key",
        api_url="url",
        rounds=2,
    )

    round_2_prompts = [p for p in prompts_seen if "round 2" in p.lower()]
    assert round_2_prompts, "expected at least one round-2 prompt"
    for prompt in round_2_prompts:
        assert "Round 2 -" not in prompt
        assert "Round 3 -" not in prompt
