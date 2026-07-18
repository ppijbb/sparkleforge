"""Issue #681: token cost/budget warnings wired into MetaState.

total_tokens_used existed in MetaState but nothing ever updated or checked
it. These tests cover the budget-check helper itself and that AgentHarness's
executor node accumulates tokens from task results and warns on overrun.
"""

from types import SimpleNamespace

from src.core.agent_harness import AgentHarness
from src.core.harness_state import check_token_budget, create_initial_harness_state


def test_check_token_budget_unlimited_when_zero():
    assert check_token_budget({"total_tokens_used": 999_999}, session_token_limit=0) is None


def test_check_token_budget_under_limit_is_none():
    assert check_token_budget({"total_tokens_used": 100}, session_token_limit=200) is None


def test_check_token_budget_at_or_over_limit_warns():
    warning = check_token_budget({"total_tokens_used": 200}, session_token_limit=200)
    assert warning is not None
    assert "200" in warning


def test_update_token_budget_accumulates_across_tasks(monkeypatch):
    monkeypatch.setattr(
        "src.core.researcher_config.get_cost_budget_config",
        lambda: SimpleNamespace(session_token_limit=0),
    )
    harness = object.__new__(AgentHarness)
    state = create_initial_harness_state("session-1", "do the thing")

    tasks = [
        {"task_id": "t1", "result": {"tokens_used": 150}},
        {"task_id": "t2", "result": {"tokens_used": 50}},
        {"task_id": "t3", "result": "no token info here"},
    ]
    harness._update_token_budget(state, tasks)

    assert state["meta"]["total_tokens_used"] == 200
    assert state["meta"]["warnings"] == []


def test_update_token_budget_warns_when_over_limit(monkeypatch):
    monkeypatch.setattr(
        "src.core.researcher_config.get_cost_budget_config",
        lambda: SimpleNamespace(session_token_limit=100),
    )
    harness = object.__new__(AgentHarness)
    state = create_initial_harness_state("session-1", "do the thing")

    harness._update_token_budget(state, [{"task_id": "t1", "result": {"tokens_used": 150}}])

    assert state["meta"]["total_tokens_used"] == 150
    assert len(state["meta"]["warnings"]) == 1
    assert "150" in state["meta"]["warnings"][0]


def test_update_token_budget_degrades_gracefully_without_config(monkeypatch):
    def _boom():
        raise RuntimeError("config unavailable")

    monkeypatch.setattr("src.core.researcher_config.get_cost_budget_config", _boom)
    harness = object.__new__(AgentHarness)
    state = create_initial_harness_state("session-1", "do the thing")

    harness._update_token_budget(state, [{"task_id": "t1", "result": {"tokens_used": 10}}])

    assert state["meta"]["total_tokens_used"] == 10
    assert state["meta"]["warnings"] == []
