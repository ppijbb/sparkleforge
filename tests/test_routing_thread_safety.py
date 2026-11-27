"""Thread-safety and dynamic model key resolution tests for RoutingMixin."""

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

from src.core.llm_manager.routing import RoutingMixin
from src.core.llm_manager.types import ModelConfig, TaskType


class _DummyRouter(RoutingMixin):
    def __init__(self, models: dict[str, ModelConfig], configured_model: str):
        self.models = models
        self.provider_rotation_order = ["google", "nvidia"]
        self.provider_rate_limited = {}
        self.provider_usage_count = {"google": 0, "nvidia": 0}
        self.llm_config = SimpleNamespace(
            budget_limit=10.0,
            primary_model=configured_model,
            planning_model=configured_model,
            reasoning_model=configured_model,
            verification_model=configured_model,
            generation_model=configured_model,
            compression_model=configured_model,
        )


def _google_model(name: str = "gemini-flash-lite", model_id: str = "gemini-3.5-flash-lite") -> ModelConfig:
    return ModelConfig(
        name=name,
        provider="google",
        model_id=model_id,
        temperature=0.1,
        max_tokens=2000,
        cost_per_token=0.0001,
        speed_rating=9.0,
        quality_rating=7.0,
        capabilities=[TaskType.PLANNING, TaskType.GENERATION],
    )


def _nvidia_model() -> ModelConfig:
    return ModelConfig(
        name="nvidia/nemotron-3-ultra-550b-a55b",
        provider="nvidia",
        model_id="nvidia/nemotron-3-ultra-550b-a55b",
        temperature=0.2,
        max_tokens=4000,
        cost_per_token=0.0,
        speed_rating=8.0,
        quality_rating=9.0,
        capabilities=list(TaskType),
    )


def test_resolve_configured_model_key_matches_existing_model_id_without_mutation():
    original_models = {"gemini-flash-lite": _google_model()}
    router = _DummyRouter(original_models, "google/gemini-3.5-flash-lite")

    selected = router.select_model(TaskType.PLANNING)

    assert selected == "gemini-flash-lite"
    assert router.models is original_models
    assert list(router.models.keys()) == ["gemini-flash-lite"]


def test_resolve_configured_model_key_copy_on_write_preserves_existing_snapshot():
    original_models = {"gemini-flash-lite": _google_model()}
    router = _DummyRouter(original_models, "google/gemini-3.1-flash-lite-preview")

    selected = router.select_model(TaskType.GENERATION)

    assert selected == "google/gemini-3.1-flash-lite-preview"
    assert router.models is not original_models
    assert "google/gemini-3.1-flash-lite-preview" not in original_models
    cfg = router.models["google/gemini-3.1-flash-lite-preview"]
    assert cfg.provider == "google"
    assert cfg.model_id == "gemini-3.1-flash-lite-preview"
    assert TaskType.GENERATION in cfg.capabilities


def test_resolve_configured_model_key_skips_dynamic_google_when_google_not_loaded():
    nvidia = _nvidia_model()
    original_models = {nvidia.name: nvidia}
    router = _DummyRouter(original_models, "google/gemini-3.1-flash-lite-preview")

    selected = router.select_model(TaskType.GENERATION)

    assert selected == nvidia.name
    assert router.models is original_models


def test_concurrent_select_model_is_safe_during_iteration():
    original_models = {"gemini-flash-lite": _google_model()}
    router = _DummyRouter(original_models, "google/gemini-3.1-flash-lite-preview")

    def worker(idx: int) -> tuple[str | None, str]:
        key = f"google/gemini-dynamic-{idx % 5}"
        direct = router._resolve_configured_model_key(key)
        selected = router.select_model(TaskType.GENERATION)
        # Iterate over the snapshot reference safely without RuntimeError.
        for name, cfg in router.models.items():
            assert cfg.provider == "google"
            assert isinstance(name, str)
        return direct, selected

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(worker, range(40)))

    assert len(results) == 40
    assert all(d and d.startswith("google/gemini-dynamic-") for d, _ in results)
    assert all(s == "google/gemini-3.1-flash-lite-preview" for _, s in results)
