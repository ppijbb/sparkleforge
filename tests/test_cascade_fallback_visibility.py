"""_try_fallback_models() must announce each attempt and the final outcome
on stdout, not just in the log file -- otherwise a failing fallback loop
looks identical to a hang from the terminal (see issue: fallback attempts
printed to stdout)."""

import asyncio

import pytest

from src.core.llm_manager.cascade import CascadeMixin
from src.core.llm_manager.types import ModelConfig, TaskType


class _FailingRegistry(CascadeMixin):
    def __init__(self):
        self.provider_rotation_order = ["google"]
        self.models = {
            "gemini-flash": ModelConfig(
                name="gemini-flash",
                provider="google",
                model_id="gemini-2.5-flash",
                temperature=0.1,
                max_tokens=100,
                cost_per_token=0.0,
                speed_rating=8.0,
                quality_rating=8.0,
                capabilities=[TaskType.GENERATION],
            ),
        }

    def _is_provider_rate_limited(self, provider):
        return False

    async def _execute_gemini_model(self, model_name, prompt, system_message=None, **kwargs):
        raise RuntimeError("simulated provider outage")


def test_fallback_prints_attempt_and_exhaustion(capsys):
    registry = _FailingRegistry()

    with pytest.raises(RuntimeError, match="All fallback models failed"):
        asyncio.run(
            registry._try_fallback_models(TaskType.GENERATION, "hello", None)
        )

    out = capsys.readouterr().out
    assert "trying fallback: gemini-flash (google)" in out
    assert "gemini-flash failed" in out
    assert "all fallback models exhausted" in out
