"""Issue #1339: the cascade tried fallback models without checking whether
the request would even fit the model's known TPM/context limit, so an
oversized request against a small-limit fallback (e.g. Groq's
openai/gpt-oss-20b free tier, TPM 8000) failed with a deterministic 413 no
retry could ever fix. The cascade should skip a model it already knows is
too small, and should learn a limit from a live 413 response so later
attempts (including retries) skip it too.
"""

import asyncio
import time

import pytest

from src.core.llm_manager.cascade import CascadeMixin
from src.core.llm_manager.types import ModelConfig, TaskType


def _model(name, **overrides):
    defaults = dict(
        name=name,
        provider="groq",
        model_id=name,
        temperature=0.1,
        max_tokens=2000,
        cost_per_token=0.0,
        speed_rating=8.0,
        quality_rating=8.0,
        capabilities=[TaskType.GENERATION],
    )
    defaults.update(overrides)
    return ModelConfig(**defaults)


class _Registry(CascadeMixin):
    def __init__(self, models, executed_log):
        self.provider_rotation_order = ["groq"]
        self.models = models
        self._executed_log = executed_log

    def _is_provider_rate_limited(self, provider):
        return False

    async def _execute_groq_model(self, model_name, prompt, system_message=None, **kwargs):
        self._executed_log.append(model_name)
        raise RuntimeError("simulated failure")


def test_oversized_request_skips_model_with_known_small_limit():
    executed = []
    registry = _Registry(
        {"small-model": _model("small-model", context_limit_tokens=100)},
        executed,
    )

    huge_prompt = "word " * 1000  # ~1300 estimated tokens, well over the 100 limit

    with pytest.raises(RuntimeError, match="All fallback models failed"):
        asyncio.run(registry._try_fallback_models(TaskType.GENERATION, huge_prompt, None))

    assert executed == [], "model with a too-small known limit must never be attempted"


def test_request_within_limit_is_attempted():
    executed = []
    registry = _Registry(
        {"small-model": _model("small-model", context_limit_tokens=100000)},
        executed,
    )

    with pytest.raises(RuntimeError, match="All fallback models failed"):
        asyncio.run(registry._try_fallback_models(TaskType.GENERATION, "hello", None))

    assert executed == ["small-model"]


def test_unknown_limit_defaults_to_attempting(capsys):
    executed = []
    registry = _Registry(
        {"unknown-limit-model": _model("unknown-limit-model", context_limit_tokens=None)},
        executed,
    )
    huge_prompt = "word " * 5000

    with pytest.raises(RuntimeError, match="All fallback models failed"):
        asyncio.run(registry._try_fallback_models(TaskType.GENERATION, huge_prompt, None))

    assert executed == ["unknown-limit-model"]


def test_413_response_is_learned():
    class _413Registry(_Registry):
        async def _execute_groq_model(self, model_name, prompt, system_message=None, **kwargs):
            self._executed_log.append(model_name)
            raise RuntimeError(
                "413 — Request too large for model in organization ... on tokens "
                "per minute (TPM): Limit 8000, Requested 10238"
            )

    executed = []
    registry = _413Registry(
        {"small-model": _model("small-model", context_limit_tokens=None)},
        executed,
    )

    with pytest.raises(RuntimeError, match="All fallback models failed"):
        asyncio.run(registry._try_fallback_models(TaskType.GENERATION, "hi", None))

    assert executed == ["small-model"]
    assert registry.models["small-model"].context_limit_tokens == 8000


def test_stale_learned_limit_is_revalidated_instead_of_skipped_forever():
    """#1349: a limit learned from a transient TPM throttle must expire, or a
    momentary rate limit permanently removes the model from the pool."""
    executed = []
    registry = _Registry(
        {
            "small-model": _model(
                "small-model",
                context_limit_tokens=100,
                context_limit_learned_at=time.time()
                - CascadeMixin.CONTEXT_LIMIT_REVALIDATION_SECONDS
                - 1,
            )
        },
        executed,
    )

    huge_prompt = "word " * 1000  # would exceed the stale 100-token limit

    with pytest.raises(RuntimeError, match="All fallback models failed"):
        asyncio.run(registry._try_fallback_models(TaskType.GENERATION, huge_prompt, None))

    assert executed == ["small-model"], "expired learned limit must not block retrying the model"


def test_fresh_learned_limit_still_skips_model():
    executed = []
    registry = _Registry(
        {
            "small-model": _model(
                "small-model", context_limit_tokens=100, context_limit_learned_at=time.time()
            )
        },
        executed,
    )

    huge_prompt = "word " * 1000

    with pytest.raises(RuntimeError, match="All fallback models failed"):
        asyncio.run(registry._try_fallback_models(TaskType.GENERATION, huge_prompt, None))

    assert executed == [], "a recently learned limit must still be honored"


def test_413_message_does_not_get_misclassified_as_generic_retry(capsys):
    class _413Registry(_Registry):
        async def _execute_groq_model(self, model_name, prompt, system_message=None, **kwargs):
            self._executed_log.append(model_name)
            raise RuntimeError("413 Request too large: tokens per minute (TPM) exceeded")

    executed = []
    registry = _413Registry(
        {"small-model": _model("small-model")},
        executed,
    )

    with pytest.raises(RuntimeError, match="All fallback models failed"):
        asyncio.run(registry._try_fallback_models(TaskType.GENERATION, "hi", None))

    out = capsys.readouterr().out
    assert "request too large for this model (413/TPM)" in out
